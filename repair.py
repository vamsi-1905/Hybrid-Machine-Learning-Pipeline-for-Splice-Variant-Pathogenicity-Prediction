import pandas as pd

# ============================================================
# CONFIG
# ============================================================

INPUT_CSV  = "splice_dataset_full.csv"
OUTPUT_CSV = "splice_dataset_ACTUALLY_balanced.csv"

RANDOM_STATE = 42

# ============================================================
# LOAD
# ============================================================

print("Loading dataset...")

df = pd.read_csv(INPUT_CSV)

df["label"] = df["label"].astype(int)

print("\nORIGINAL COUNTS:")
print(df["label"].value_counts())

# ============================================================
# SPLIT CLASSES
# ============================================================

benign = df[df["label"] == 0].copy()
path   = df[df["label"] == 1].copy()

n_ben  = len(benign)
n_path = len(path)

print(f"\nBenign     : {n_ben:,}")
print(f"Pathogenic : {n_path:,}")

# ============================================================
# OVERSAMPLE BENIGN
# ============================================================

needed = n_path - n_ben

print(f"\nNeed {needed:,} more benign rows")

extra_benign = benign.sample(
    n=needed,
    replace=True,
    random_state=RANDOM_STATE
)

balanced = pd.concat(
    [path, benign, extra_benign],
    ignore_index=True
)

# ============================================================
# SHUFFLE
# ============================================================

balanced = balanced.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

# ============================================================
# VERIFY
# ============================================================

print("\nFINAL COUNTS:")
print(balanced["label"].value_counts())

# ============================================================
# SAVE
# ============================================================

balanced.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved -> {OUTPUT_CSV}")

print("\nDONE.")