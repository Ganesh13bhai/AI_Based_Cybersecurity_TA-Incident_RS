import json
import glob
import os
from collections import defaultdict

import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "metadata",
    "label_mapping.json"
)


def get_dataset_name(path):
    path_lower = path.lower()

    if "cicids2017" in path_lower:
        return "CIC-IDS2017"

    if "cicids2018" in path_lower:
        return "CIC-IDS2018"

    if "unswnb15" in path_lower:
        return "UNSW-NB15"

    return "UNKNOWN"


def get_original_label_column(df, dataset):
    if dataset == "UNSW-NB15":
        if "attack_cat" in df.columns:
            return "attack_cat"

    if "Label" in df.columns:
        return "Label"

    if "label" in df.columns and "canonical_label" not in df.columns:
        return "label"

    return None


def main():

    files = sorted(
        glob.glob(
            os.path.join(PROCESSED_DIR, "**", "*.csv"),
            recursive=True
        )
    )

    files = [
        f for f in files
        if not os.path.basename(f).startswith("._")
    ]

    if not files:
        raise FileNotFoundError(
            f"No processed CSV files found in {PROCESSED_DIR}"
        )

    mapping = defaultdict(dict)
    errors = []
    total_rows = 0

    print("=" * 70)
    print("CyberSecAI - Label Mapping Generator")
    print("=" * 70)

    print(f"\nProcessed CSV files found: {len(files)}")

    for file in files:

        dataset = get_dataset_name(file)

        if dataset == "UNKNOWN":
            print(f"\nWARNING: Unknown dataset: {file}")
            continue

        print(f"\nProcessing: {dataset}")
        print(f"File: {os.path.basename(file)}")

        try:
            df = pd.read_csv(file, usecols=lambda c: c in {
                "Label",
                "label",
                "attack_cat",
                "canonical_label",
                "binary_label"
            })
        except Exception as e:
            errors.append({
                "file": file,
                "error": str(e)
            })
            continue

        required_columns = {"canonical_label", "binary_label"}

        missing = required_columns - set(df.columns)

        if missing:
            errors.append({
                "file": file,
                "error": f"Missing required columns: {sorted(missing)}"
            })
            continue

        original_col = get_original_label_column(df, dataset)

        if original_col is None:
            errors.append({
                "file": file,
                "error": "Could not identify original label column"
            })
            continue

        total_rows += len(df)

        subset = df[
            [original_col, "canonical_label", "binary_label"]
        ].drop_duplicates()

        print(f"Original label column: {original_col}")
        print(f"Unique label mappings found: {len(subset)}")

        for _, row in subset.iterrows():

            original = str(row[original_col]).strip()
            canonical = str(row["canonical_label"]).strip()

            try:
                binary = int(float(row["binary_label"]))
            except Exception:
                errors.append({
                    "file": file,
                    "error": (
                        f"Invalid binary label for "
                        f"'{original}': {row['binary_label']}"
                    )
                })
                continue

            if not original or original.lower() == "nan":
                continue

            if canonical.lower() == "nan":
                errors.append({
                    "file": file,
                    "error": f"Missing canonical label for '{original}'"
                })
                continue

            if binary not in (0, 1):
                errors.append({
                    "file": file,
                    "error": (
                        f"Binary label must be 0 or 1 for "
                        f"'{original}', got {binary}"
                    )
                })
                continue

            # Check for inconsistent mapping
            if original in mapping[dataset]:

                existing = mapping[dataset][original]

                if (
                    existing["canonical"] != canonical
                    or existing["binary"] != binary
                ):
                    errors.append({
                        "dataset": dataset,
                        "label": original,
                        "error": (
                            "Inconsistent mapping: "
                            f"{existing} vs "
                            f"canonical={canonical}, binary={binary}"
                        )
                    })

            else:
                mapping[dataset][original] = {
                    "canonical": canonical,
                    "binary": binary
                }

    # Convert defaultdict to normal dict
    mapping = {
        dataset: dict(labels)
        for dataset, labels in mapping.items()
    }

    output = {
        "version": "1.0",
        "description": (
            "Explicit mapping from original dataset labels to "
            "CyberSecAI canonical and binary labels."
        ),
        "source": (
            "Derived from processed CSV files containing "
            "canonical_label and binary_label."
        ),
        "datasets": mapping,
        "validation": {
            "processed_file_count": len(files),
            "processed_row_count": total_rows,
            "mapping_error_count": len(errors),
            "status": "PASS" if not errors else "FAIL"
        },
        "errors": errors
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("LABEL MAPPING SUMMARY")
    print("=" * 70)

    for dataset, labels in mapping.items():

        print(f"\n{dataset}")
        print("-" * 40)

        for original, info in labels.items():
            print(
                f"{original} "
                f"-> {info['canonical']} "
                f"-> {info['binary']}"
            )

    print("\n" + "=" * 70)
    print(f"Total processed rows: {total_rows:,}")
    print(f"Mapping errors: {len(errors)}")
    print(f"Status: {'PASS' if not errors else 'FAIL'}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("=" * 70)

    if errors:
        print("\nERROR DETAILS:")
        for error in errors:
            print(error)


if __name__ == "__main__":
    main()