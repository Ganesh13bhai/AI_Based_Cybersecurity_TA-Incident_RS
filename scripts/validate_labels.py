import json
import glob
import os

import pandas as pd


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PROCESSED_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

MAPPING_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "metadata",
    "label_mapping.json"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "metadata",
    "label_validation.json"
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

    if "label" in df.columns:
        return "label"

    return None


def main():

    print("=" * 70)
    print("CyberSecAI - Label Validation")
    print("=" * 70)

    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)

    mappings = mapping_data["datasets"]

    files = sorted(
        glob.glob(
            os.path.join(
                PROCESSED_DIR,
                "**",
                "*.csv"
            ),
            recursive=True
        )
    )

    files = [
        f for f in files
        if not os.path.basename(f).startswith("._")
    ]

    results = []

    total_rows = 0
    total_errors = 0
    total_unknown_labels = 0

    for file in files:

        dataset = get_dataset_name(file)

        if dataset == "UNKNOWN":
            continue

        print(f"\nChecking: {dataset}")
        print(f"File: {os.path.basename(file)}")

        df = pd.read_csv(file)

        rows = len(df)
        total_rows += rows

        errors = []

        # --------------------------------------------------
        # Required columns
        # --------------------------------------------------

        required = {
            "canonical_label",
            "binary_label"
        }

        missing_columns = required - set(df.columns)

        if missing_columns:

            errors.append(
                f"Missing columns: {sorted(missing_columns)}"
            )

            results.append({
                "dataset": dataset,
                "file": os.path.basename(file),
                "rows": rows,
                "status": "FAIL",
                "errors": errors
            })

            total_errors += len(errors)
            continue

        original_column = get_original_label_column(
            df,
            dataset
        )

        if original_column is None:

            errors.append(
                "Original label column not found"
            )

            results.append({
                "dataset": dataset,
                "file": os.path.basename(file),
                "rows": rows,
                "status": "FAIL",
                "errors": errors
            })

            total_errors += len(errors)
            continue

        # --------------------------------------------------
        # Null checks
        # --------------------------------------------------

        null_original = int(
            df[original_column].isna().sum()
        )

        null_canonical = int(
            df["canonical_label"].isna().sum()
        )

        null_binary = int(
            df["binary_label"].isna().sum()
        )

        if null_original:
            errors.append(
                f"Original label nulls: {null_original}"
            )

        if null_canonical:
            errors.append(
                f"Canonical label nulls: {null_canonical}"
            )

        if null_binary:
            errors.append(
                f"Binary label nulls: {null_binary}"
            )

        # --------------------------------------------------
        # Binary label validation
        # --------------------------------------------------

        binary_values = set(
            pd.to_numeric(
                df["binary_label"],
                errors="coerce"
            ).dropna().astype(int).unique()
        )

        invalid_binary = binary_values - {0, 1}

        if invalid_binary:

            errors.append(
                f"Invalid binary labels: "
                f"{sorted(invalid_binary)}"
            )

        # --------------------------------------------------
        # Original -> canonical mapping validation
        # --------------------------------------------------

        dataset_mapping = mappings.get(
            dataset,
            {}
        )

        unknown_labels = set()

        inconsistent_rows = 0

        for original in df[original_column].dropna().unique():

            original = str(original).strip()

            if original not in dataset_mapping:

                unknown_labels.add(original)
                continue

            expected = dataset_mapping[original]

            actual_rows = df[
                df[original_column].astype(str).str.strip()
                == original
            ]

            canonical_values = set(
                actual_rows[
                    "canonical_label"
                ].astype(str).str.strip().unique()
            )

            binary_values = set(
                pd.to_numeric(
                    actual_rows["binary_label"],
                    errors="coerce"
                ).dropna().astype(int).unique()
            )

            if canonical_values != {
                expected["canonical"]
            }:

                inconsistent_rows += len(actual_rows)

            if binary_values != {
                expected["binary"]
            }:

                inconsistent_rows += len(actual_rows)

        if unknown_labels:

            total_unknown_labels += len(
                unknown_labels
            )

            errors.append(
                "Unknown original labels: "
                + ", ".join(
                    sorted(unknown_labels)
                )
            )

        if inconsistent_rows:

            errors.append(
                f"Inconsistent mapping rows: "
                f"{inconsistent_rows}"
            )

        # --------------------------------------------------
        # Class distribution
        # --------------------------------------------------

        canonical_distribution = (
            df["canonical_label"]
            .value_counts()
            .to_dict()
        )

        binary_distribution = (
            df["binary_label"]
            .value_counts()
            .to_dict()
        )

        status = "PASS" if not errors else "FAIL"

        if errors:
            total_errors += len(errors)

        results.append({
            "dataset": dataset,
            "file": os.path.basename(file),
            "rows": rows,
            "original_label_column": original_column,
            "canonical_classes": sorted(
                df["canonical_label"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            ),
            "canonical_distribution": {
                str(k): int(v)
                for k, v in canonical_distribution.items()
            },
            "binary_distribution": {
                str(k): int(v)
                for k, v in binary_distribution.items()
            },
            "unknown_labels": sorted(
                unknown_labels
            ),
            "status": status,
            "errors": errors
        })

        print(f"Rows: {rows:,}")
        print(f"Status: {status}")

        if errors:
            for error in errors:
                print(f"  ERROR: {error}")

    overall_status = (
        "PASS"
        if total_errors == 0
        else "FAIL"
    )

    output = {
        "version": "1.0",
        "description": (
            "Independent validation of label consistency "
            "across all processed CyberSecAI datasets."
        ),
        "summary": {
            "processed_file_count": len(results),
            "processed_row_count": total_rows,
            "unknown_label_count": total_unknown_labels,
            "error_count": total_errors,
            "status": overall_status
        },
        "files": results
    }

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 70)
    print("FINAL LABEL VALIDATION")
    print("=" * 70)

    print(
        f"Files checked       : {len(results)}"
    )

    print(
        f"Rows checked        : {total_rows:,}"
    )

    print(
        f"Unknown labels      : {total_unknown_labels}"
    )

    print(
        f"Validation errors   : {total_errors}"
    )

    print(
        f"Overall status      : {overall_status}"
    )

    print(
        f"Saved to            : {OUTPUT_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()