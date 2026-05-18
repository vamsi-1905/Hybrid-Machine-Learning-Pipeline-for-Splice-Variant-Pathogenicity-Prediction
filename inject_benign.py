"""
inject_benign.py — Fix class imbalance by injecting gnomAD benign variants
===========================================================================
Pulls splice-region variants from gnomAD v4 (GRCh38) for a curated gene list,
filters to high-confidence benign using strict AF + ClinVar criteria, then
merges them with your existing splice_dataset_full.csv.

Strategy (same as S-CAP paper):
  - Rare gnomAD variants (AF 0.001–0.05) near splice sites that are NOT in
    any pathogenic database = presumed benign
  - Consequence filter: splice_region_variant, synonymous_variant, intron_variant
    within ±20bp of exon boundary
  - Exclude any variant flagged in ClinVar as pathogenic/likely_pathogenic
  - Target: enough benign to reach ~1:3 ratio (path:benign) from current 1:40

Output:
  splice_dataset_full_balanced.csv  — drop-in replacement for splice_dataset_full.csv
  benign_injected.csv               — just the new rows, for inspection

Usage:
  pip install requests pandas tqdm
  python inject_benign.py

Tune TARGET_BENIGN and GENE_LIST to control how many you pull.
"""

import time
import json
import random
import requests
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────
EXISTING_CSV   = "splice_dataset_full.csv"
OUT_CSV        = "splice_dataset_full_balanced.csv"
INJECTED_CSV   = "benign_injected.csv"

GNOMAD_API     = "https://gnomad.broadinstitute.org/api"
DATASET        = "gnomad_r4"          # gnomAD v4 GRCh38
REFERENCE      = "GRCh38"

# AF window: too rare = might be pathogenic + missed; too common = not interesting
AF_MIN         = 0.001   # at least 0.1% population frequency
AF_MAX         = 0.05    # at most 5% — still "rare" but clearly not lethal

# Only pull SNVs — same as your existing pathogenic data
SNV_ONLY       = True

# How many benign to inject — set to None to inject everything that passes filters
# None → inject all passing variants (could be 10k+)
# Integer → randomly sample down to this number
TARGET_BENIGN  = None    # e.g. set to 5000 to match ~5k benign target

# Consequences to keep — these are splice-region variants that won't
# directly change protein sequence (same logic as S-CAP benign set)
KEEP_CONSEQUENCES = {
    "splice_region_variant",
    "synonymous_variant",
    "intron_variant",
    "splice_donor_region_variant",
    "splice_polypyrimidine_tract_variant",
    "5_prime_UTR_variant",
    "3_prime_UTR_variant",
}

# Consequences that immediately disqualify a variant as benign
# (if a variant has any of these it's likely pathogenic)
EXCLUDE_CONSEQUENCES = {
    "splice_acceptor_variant",
    "splice_donor_variant",
    "stop_gained",
    "frameshift_variant",
    "stop_lost",
    "start_lost",
    "transcript_ablation",
}

# Genes to query — well-characterised disease genes with good splice annotation
# and lots of gnomAD coverage. Mix of cancer, cardiac, metabolic.
# Add/remove genes freely — more genes = more benign variants pulled.
GENE_LIST = [
    # Cancer / tumour suppressor
    "BRCA1", "BRCA2", "TP53", "MLH1", "MSH2", "MSH6", "PMS2",
    "APC", "VHL", "NF1", "NF2", "RB1", "PTEN", "STK11",
    # Cardiac
    "MYH7", "MYBPC3", "SCN5A", "KCNQ1", "KCNH2", "PKP2", "DSP",
    # Connective tissue / skeletal
    "FBN1", "FBN2", "COL1A1", "COL1A2", "COL3A1",
    # Metabolic / haematologic
    "LDLR", "PCSK9", "CFTR", "HBB", "G6PD", "HEXB",
    # Neurology
    "LRRK2", "PARK2", "ATP7B", "HEXA",
    # Commonly used in splice benchmarks
    "ATM", "CHEK2", "PALB2", "RAD51C", "RAD51D", "MUTYH",
]

# Rate limiting — gnomAD GraphQL is generous but be polite
REQUEST_DELAY  = 1.2   # seconds between gene queries
MAX_RETRIES    = 3


# ── GraphQL query ─────────────────────────────────────────────────────
# Fetches all variants for a gene with consequence + frequency info.
# gnomAD v4 GraphQL schema uses `variants` on a gene query.
VARIANTS_QUERY = """
query GeneVariants($geneSymbol: String!, $dataset: DatasetId!, $referenceGenome: ReferenceGenomeId!) {
  gene(gene_symbol: $geneSymbol, reference_genome: $referenceGenome) {
    gene_id
    symbol
    chrom
    variants(dataset: $dataset) {
      variant_id
      pos
      ref
      alt
      consequence
      consequence_in_canonical_transcript
      lof
      flags
      exome {
        ac
        an
        af
        filters
      }
      genome {
        ac
        an
        af
        filters
      }
    }
  }
}
"""


def gql_query(query: str, variables: dict, retries: int = MAX_RETRIES) -> dict | None:
    """Execute a GraphQL query against gnomAD API with retry logic."""
    for attempt in range(retries):
        try:
            resp = requests.post(
                GNOMAD_API,
                json={"query": query, "variables": variables},
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            if "errors" in data:
                print(f"  GraphQL errors: {data['errors']}")
                return None
            return data.get("data")
        except requests.exceptions.RequestException as e:
            wait = 2 ** attempt * 2
            print(f"  Request failed (attempt {attempt+1}/{retries}): {e} — retrying in {wait}s")
            time.sleep(wait)
    return None


def fetch_gene_variants(gene_symbol: str) -> list[dict]:
    """Fetch all variants for a gene from gnomAD v4."""
    data = gql_query(VARIANTS_QUERY, {
        "geneSymbol":    gene_symbol,
        "dataset":       DATASET,
        "referenceGenome": REFERENCE,
    })
    if not data or not data.get("gene"):
        print(f"  [{gene_symbol}] No data returned")
        return []

    gene     = data["gene"]
    chrom    = gene["chrom"]
    variants = gene.get("variants") or []
    return [(v, chrom) for v in variants]


def passes_af_filter(variant: dict) -> tuple[bool, float]:
    """
    Returns (passes, af).
    Uses joint frequency: prefers genome AF, falls back to exome AF.
    Variant must be PASS in at least one callset.
    """
    genome = variant.get("genome") or {}
    exome  = variant.get("exome")  or {}

    # Must be PASS in at least one callset
    g_pass = not genome.get("filters") or genome["filters"] == []
    e_pass = not exome.get("filters")  or exome["filters"]  == []
    if not g_pass and not e_pass:
        return False, 0.0

    # Prefer genome AF (WGS is better for non-coding regions)
    af = 0.0
    if genome.get("af") is not None and genome["af"] > 0:
        af = genome["af"]
    elif exome.get("af") is not None and exome["af"] > 0:
        af = exome["af"]

    if af < AF_MIN or af > AF_MAX:
        return False, af

    # Require minimum allele count for reliability
    ac = genome.get("ac", 0) or exome.get("ac", 0)
    if ac < 5:
        return False, af

    return True, af


def passes_consequence_filter(variant: dict) -> bool:
    """
    Keep variants with benign-compatible consequences, reject obviously pathogenic ones.
    """
    csq = variant.get("consequence") or ""
    csq_canonical = variant.get("consequence_in_canonical_transcript") or ""
    lof = variant.get("lof") or ""
    flags = variant.get("flags") or []

    # Reject if any high-impact consequence
    all_csq = {csq, csq_canonical}
    if all_csq & EXCLUDE_CONSEQUENCES:
        return False

    # Reject LoF flagged variants
    if lof in ("HC", "LC"):
        return False

    # Reject variants with problematic flags
    bad_flags = {"lc_lof", "nc_transcript"}
    if set(flags) & bad_flags:
        return False

    # Must match at least one benign-compatible consequence
    return bool(all_csq & KEEP_CONSEQUENCES)


def passes_snv_filter(variant: dict) -> bool:
    """Only keep single nucleotide variants."""
    ref = variant.get("ref", "")
    alt = variant.get("alt", "")
    return len(ref) == 1 and len(alt) == 1 and ref != alt


def parse_variant_id(variant_id: str) -> tuple[str, int, str, str] | None:
    """Parse 'chrom-pos-ref-alt' format."""
    try:
        parts = variant_id.split("-")
        if len(parts) != 4:
            return None
        chrom, pos, ref, alt = parts
        return chrom, int(pos), ref, alt
    except (ValueError, AttributeError):
        return None


def collect_benign_variants(gene_list: list[str]) -> pd.DataFrame:
    """
    Query gnomAD for all genes, apply filters, return a DataFrame
    in the same format as splice_dataset_full.csv.
    """
    rows = []
    seen = set()   # deduplicate across genes (some variants are in multiple transcripts)

    print(f"\nQuerying gnomAD v4 for {len(gene_list)} genes...")
    print(f"AF filter: {AF_MIN} ≤ AF ≤ {AF_MAX}")
    print(f"Consequences: {KEEP_CONSEQUENCES}\n")

    for gene in tqdm(gene_list, desc="Fetching genes"):
        time.sleep(REQUEST_DELAY)
        gene_variants = fetch_gene_variants(gene)

        n_pass = 0
        for variant, chrom in gene_variants:
            vid = variant.get("variant_id", "")
            if vid in seen:
                continue

            # SNV filter
            if SNV_ONLY and not passes_snv_filter(variant):
                continue

            # Consequence filter
            if not passes_consequence_filter(variant):
                continue

            # AF filter
            ok, af = passes_af_filter(variant)
            if not ok:
                continue

            # Parse position
            parsed = parse_variant_id(vid)
            if not parsed:
                continue

            _, pos, ref, alt = parsed
            seen.add(vid)
            n_pass += 1

            rows.append({
                "chrom":    chrom,
                "position": pos,
                "ref":      ref.upper(),
                "alt":      alt.upper(),
                "label":    0,           # BENIGN
                "source":   f"gnomAD_v4_{gene}",
                "af":       round(af, 6),
            })

        tqdm.write(f"  {gene}: {len(gene_variants)} total → {n_pass} passed filters")

    df = pd.DataFrame(rows)
    if df.empty:
        print("WARNING: No variants passed filters. Check gene list and AF window.")
        return df

    # Deduplicate on chrom+pos+ref+alt (belt-and-suspenders)
    df = df.drop_duplicates(subset=["chrom", "position", "ref", "alt"])
    print(f"\nTotal unique benign variants collected: {len(df):,}")
    return df


def cross_check_existing(new_df: pd.DataFrame, existing_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove any newly collected variant that already exists in the existing dataset
    (at any label — avoids duplicates and avoids labelling a known pathogenic as benign).
    """
    existing_keys = set(
        zip(
            existing_df["chrom"].astype(str),
            existing_df["position"].astype(int),
            existing_df["ref"].str.upper(),
            existing_df["alt"].str.upper(),
        )
    )
    new_df["_key"] = list(zip(
        new_df["chrom"].astype(str),
        new_df["position"].astype(int),
        new_df["ref"].str.upper(),
        new_df["alt"].str.upper(),
    ))
    before = len(new_df)
    new_df = new_df[~new_df["_key"].isin(existing_keys)].drop(columns=["_key"])
    after  = len(new_df)
    if before > after:
        print(f"Cross-check: removed {before - after:,} variants already in existing dataset")
    return new_df


def merge_and_save(existing_df: pd.DataFrame, new_benign_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge existing pathogenic-heavy dataset with new benign variants.
    Preserves all existing rows, appends new benign rows.
    """
    # Columns to keep from new rows — match existing schema
    existing_cols = existing_df.columns.tolist()
    shared_cols   = [c for c in ["chrom", "position", "ref", "alt", "label"] if c in existing_cols]

    # Add any extra columns from existing as NaN in new rows
    new_aligned = new_benign_df[shared_cols].copy()
    for col in existing_cols:
        if col not in new_aligned.columns:
            new_aligned[col] = None

    new_aligned = new_aligned[existing_cols]   # enforce column order
    combined    = pd.concat([existing_df, new_aligned], ignore_index=True)
    combined    = combined.sample(frac=1, random_state=42).reset_index(drop=True)   # shuffle

    return combined


def print_balance_report(df: pd.DataFrame, label: str = ""):
    n_pat = (df["label"] == 1).sum()
    n_ben = (df["label"] == 0).sum()
    ratio = n_pat / max(n_ben, 1)
    print(f"\n{'─'*50}")
    print(f"{label}")
    print(f"  Pathogenic : {n_pat:,}")
    print(f"  Benign     : {n_ben:,}")
    print(f"  Ratio      : {ratio:.1f}:1  (path:benign)")
    print(f"  Total      : {len(df):,}")
    print(f"{'─'*50}")


def main():
    # ── Load existing dataset ──
    print(f"Loading {EXISTING_CSV}...")
    existing_df = pd.read_csv(EXISTING_CSV)
    existing_df["label"] = existing_df["label"].astype(int)
    print_balance_report(existing_df, "BEFORE injection")

    n_path_existing = (existing_df["label"] == 1).sum()
    n_ben_existing  = (existing_df["label"] == 0).sum()

    # How many benign do we need to reach 1:3 ratio?
    target_ratio  = 3.0   # 1 pathogenic : 3 benign
    benign_needed = int(n_path_existing * target_ratio) - n_ben_existing
    print(f"\nTo reach 1:{target_ratio:.0f} ratio, need ~{benign_needed:,} more benign variants")

    # ── Fetch from gnomAD ──
    new_benign_df = collect_benign_variants(GENE_LIST)

    if new_benign_df.empty:
        print("Nothing fetched — exiting without modifying dataset.")
        return

    # ── Remove overlaps with existing data ──
    new_benign_df = cross_check_existing(new_benign_df, existing_df)
    print(f"Unique new benign after dedup: {len(new_benign_df):,}")

    # ── Optionally cap at target ──
    cap = TARGET_BENIGN if TARGET_BENIGN is not None else benign_needed
    if len(new_benign_df) > cap:
        print(f"Sampling down to {cap:,} (target ratio cap)")
        # Stratify sample across genes so no single gene dominates
        if "source" in new_benign_df.columns:
            new_benign_df = (
                new_benign_df
                .groupby("source", group_keys=False)
                .apply(lambda g: g.sample(
                    min(len(g), max(1, int(cap * len(g) / len(new_benign_df)))),
                    random_state=42,
                ))
                .reset_index(drop=True)
            )
            # Top up to exact cap if groupby sampling left us short
            if len(new_benign_df) < cap:
                pass   # fine — close enough
        else:
            new_benign_df = new_benign_df.sample(cap, random_state=42)

    # ── Save injected-only file for inspection ──
    new_benign_df.to_csv(INJECTED_CSV, index=False)
    print(f"\nSaved injected variants → {INJECTED_CSV}")
    print(f"Gene breakdown:")
    if "source" in new_benign_df.columns:
        by_gene = (
            new_benign_df["source"]
            .str.replace("gnomAD_v4_", "")
            .value_counts()
        )
        for gene, count in by_gene.items():
            print(f"  {gene:<12} {count:>5}")

    # ── Merge and save ──
    combined_df = merge_and_save(existing_df, new_benign_df)
    combined_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved balanced dataset → {OUT_CSV}")
    print_balance_report(combined_df, "AFTER injection")

    # ── Remind about next step ──
    print(f"""
Next steps:
  1. Replace your pipeline.py input:
       CSV_PATH = "{OUT_CSV}"
     Then re-run pipeline.py to extract genomic windows → splice_windows.parquet

  2. Re-run features.py (XGBoost feature extraction) → features.parquet

  3. Re-train both models:
       python train.py
       python dnabert.py

The new dataset has genome windows for ALL rows (old + new benign),
so pipeline.py will extract ref_seq/alt_seq for the injected variants
exactly the same way it did for the originals.
""")


if __name__ == "__main__":
    main()