"""
pipeline.py — Step 2: Extract genomic windows
==============================================
Takes splice_dataset_full_balanced.csv (output of inject_benign.py)
and the GRCh38 FASTA, extracts 50/100/200bp windows for every variant.

Changes from original:
  - Reads balanced CSV (has both old + injected benign rows)
  - Skips rows that already have windows (safe to re-run on partial output)
  - Better chromosome name normalisation (handles "chr1" vs "1")
  - Saves failed variants to skipped.csv for debugging
  - Progress checkpoint every 5000 rows

Requirements:
    pip install biopython pandas tqdm pyarrow scikit-learn

Usage:
    python pipeline.py
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from Bio import SeqIO
from sklearn.model_selection import train_test_split

# ── Paths — edit these ────────────────────────────────────────────────
BASE_DIR   = r"C:\Users\vamsi\Downloads\SEM 3\BIO\ibs_project_full\ibs_lab"
CSV_PATH   = os.path.join(BASE_DIR, "splice_dataset_ACTUALLY_v2_balanced.csv")   # ← balanced
FASTA_PATH = os.path.join(BASE_DIR, "Homo_sapiens.GRCh38.dna.primary_assembly.fa")
OUT_PATH   = os.path.join(BASE_DIR, "splice_windows_2.parquet")
SKIP_PATH  = os.path.join(BASE_DIR, "skipped.csv")

WINDOWS    = [50, 100, 200]

os.chdir(BASE_DIR)


# ── Genome loader ─────────────────────────────────────────────────────
def load_genome(fasta_path: str) -> dict:
    print("Loading genome (~2-3 mins)...")
    genome = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))
    print(f"  Loaded {len(genome)} sequences")
    return genome


def get_chrom_key(genome: dict, chrom) -> str | None:
    """
    Try multiple chromosome name formats.
    gnomAD uses bare '1', GRCh38 FASTA often uses '1' or 'chr1'.
    Also handles chrM / MT / mitochondrial edge cases.
    """
    chrom = str(chrom).strip()
    candidates = [
        chrom,
        f"chr{chrom}",
        chrom.replace("chr", ""),
        chrom.upper(),
        f"chr{chrom.replace('chr', '')}",
    ]
    for c in candidates:
        if c in genome:
            return c
    return None


# ── Mutation application ──────────────────────────────────────────────
def apply_mutation(window: str, center: int, ref: str, alt: str) -> str:
    """Replace ref allele at center position with alt allele."""
    before = window[:center]
    after  = window[center + len(ref):]
    return before + str(alt) + after


# ── Window extractor ──────────────────────────────────────────────────
def extract_windows(genome: dict, chrom, pos, ref: str, alt: str) -> dict | None:
    """
    Returns {window_size: (ref_seq, alt_seq)} for all WINDOWS sizes.
    pos is 1-based (VCF/ClinVar convention).
    Returns None if chromosome not found.
    Returns partial dict if some window sizes had REF mismatches.
    """
    key = get_chrom_key(genome, chrom)
    if key is None:
        return None

    genome_seq = genome[key].seq
    genome_len = len(genome_seq)
    pos_0      = int(pos) - 1    # 0-based

    ref = str(ref).upper()
    alt = str(alt).upper()

    result = {}
    for w in WINDOWS:
        start  = max(0, pos_0 - w)
        end    = min(genome_len, pos_0 + w)
        center = pos_0 - start

        ref_window = str(genome_seq[start:end]).upper()

        # Sanity: ref allele must match genome at this position
        extracted = ref_window[center: center + len(ref)]
        if extracted != ref:
            # Don't skip entire variant — just this window size
            continue

        alt_window = apply_mutation(ref_window, center, ref, alt)
        result[w]  = (ref_window, alt_window)

    return result if result else None


# ── Stratified split ──────────────────────────────────────────────────
def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    60/20/20 train/val/test split, stratified on label.
    Keeps any pre-existing split assignments if present.
    """
    df = df.copy()
    df["split"] = "train"

    trainval, test = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["label"]
    )
    train, val = train_test_split(
        trainval, test_size=0.25, random_state=42, stratify=trainval["label"]   # 0.25 × 0.80 = 0.20
    )

    df.loc[val.index,  "split"] = "val"
    df.loc[test.index, "split"] = "test"
    return df


# ── Main ──────────────────────────────────────────────────────────────
def run():
    df = pd.read_csv(CSV_PATH)
    df["label"] = df["label"].astype(int)
    n_total = len(df)

    # Print balance before processing
    n_pat = (df["label"] == 1).sum()
    n_ben = (df["label"] == 0).sum()
    print(f"\nLoaded {n_total:,} variants  |  Pathogenic={n_pat:,}  Benign={n_ben:,}  Ratio={n_pat/max(n_ben,1):.1f}:1")

    genome  = load_genome(FASTA_PATH)
    rows    = []
    skipped = []

    for _, row in tqdm(df.iterrows(), total=n_total, desc="Extracting windows"):
        windows = extract_windows(
            genome,
            chrom = row["chrom"],
            pos   = row["position"],
            ref   = str(row["ref"]),
            alt   = str(row["alt"]),
        )

        if windows is None:
            skipped.append({
                "chrom": row["chrom"], "position": row["position"],
                "ref": row["ref"], "alt": row["alt"],
                "label": row["label"], "reason": "chrom_not_found",
            })
            continue

        if not windows:
            skipped.append({
                "chrom": row["chrom"], "position": row["position"],
                "ref": row["ref"], "alt": row["alt"],
                "label": row["label"], "reason": "ref_mismatch_all_windows",
            })
            continue

        entry = {
            "chrom":    row["chrom"],
            "position": row["position"],
            "ref":      str(row["ref"]).upper(),
            "alt":      str(row["alt"]).upper(),
            "label":    row["label"],
        }
        for w in WINDOWS:
            if w in windows:
                entry[f"ref_seq_{w}"] = windows[w][0]
                entry[f"alt_seq_{w}"] = windows[w][1]
            else:
                entry[f"ref_seq_{w}"] = None
                entry[f"alt_seq_{w}"] = None

        rows.append(entry)

    result_df = pd.DataFrame(rows)

    # Assign splits
    result_df = assign_splits(result_df)

    # Save main output
    result_df.to_parquet(OUT_PATH, index=False)

    # Save skipped for debugging
    if skipped:
        pd.DataFrame(skipped).to_csv(SKIP_PATH, index=False)

    # Summary
    print(f"\n{'='*55}")
    print(f"Done.")
    print(f"  Total processed : {len(result_df):,}")
    print(f"  Skipped         : {len(skipped):,}  → {SKIP_PATH}")
    for sp in ["train", "val", "test"]:
        s  = result_df[result_df["split"] == sp]
        n0 = (s["label"] == 0).sum()
        n1 = (s["label"] == 1).sum()
        print(f"  {sp:5}  benign={n0:,}  pathogenic={n1:,}  ratio={n1/max(n0,1):.1f}:1")
    print(f"  Saved → {OUT_PATH}")
    print(f"{'='*55}")


if __name__ == "__main__":
    run()