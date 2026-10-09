from pathlib import Path
import hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLIT_ROOT = ROOT / "data" / "splits" / "unswnb15"

TARGET_COLUMNS = {
    "label",
    "attack_cat",
    "canonical_label",
    "binary_label",
    "original_label",
}

def normalize_value(value):
    if pd.isna(value):
        return "<NA>"

    if isinstance(value, float):
        return f"{value:.12g}"

    return str(value).strip()


def fingerprint_row(row, feature_columns):
    payload = "\x1f".join(
        normalize_value(row[col])
        for col in feature_columns
    )

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_groups(split_name):
    split_dir = SPLIT_ROOT / split_name
    files = sorted(split_dir.glob("part_*.csv"))

    groups = set()
    row_count = 0

    feature_columns = None

    for file in files:
        df = pd.read_csv(file)

        if feature_columns is None:
            feature_columns = [
                c for c in df.columns
                if c not in TARGET_COLUMNS
            ]

        row_count += len(df)

        for _, row in df.iterrows():
            groups.add(
                fingerprint_row(row, feature_columns)
            )

    return groups, row_count


print("=" * 70)
print("UNSW-NB15 SPLIT BOUNDARY VALIDATION")
print("=" * 70)

train_groups, train_rows = get_groups("train")
val_groups, val_rows = get_groups("validation")
test_groups, test_rows = get_groups("test")

train_val = train_groups & val_groups
train_test = train_groups & test_groups
val_test = val_groups & test_groups

print()
print("Rows:")
print(f"  Train:      {train_rows:,}")
print(f"  Validation: {val_rows:,}")
print(f"  Test:       {test_rows:,}")

print()
print("Unique feature groups:")
print(f"  Train:      {len(train_groups):,}")
print(f"  Validation: {len(val_groups):,}")
print(f"  Test:       {len(test_groups):,}")

print()
print("Train/Validation overlap:", len(train_val))
print("Train/Test overlap:", len(train_test))
print("Validation/Test overlap:", len(val_test))

print()
print("Expected rows:")
print("  Train + Validation = 96,822")
print("  Test               = 49,971")

rows_ok = (
    train_rows == 82_299
    and val_rows == 14_523
    and test_rows == 49_971
)

train_val_ok = len(train_val) == 0

print()
print("Official test boundary preserved:", "PASS" if test_rows == 49_971 else "FAIL")
print("Train/Validation duplicate isolation:",
      "PASS" if train_val_ok else "FAIL")

print()
print("Known official-boundary duplicate diagnostics:")
print(f"  Train/Test overlap:      {len(train_test):,}")
print(f"  Validation/Test overlap: {len(val_test):,}")

print()
print("=" * 70)

if rows_ok and train_val_ok:
    print("UNSW-NB15 SPLIT VALIDATION: PASS")
    print("=" * 70)
else:
    print("UNSW-NB15 SPLIT VALIDATION: FAIL")
    print("=" * 70)