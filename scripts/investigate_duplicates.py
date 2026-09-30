from pathlib import Path
import json

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = Path(
    "/Volumes/Ganesh's SD/Cybersecurity-Project/datasets/raw"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "metadata"
    / "duplicate_investigation.json"
)


# ============================================================
# DATASETS
# ============================================================

DATASET_PATHS = {
    "CIC-IDS2017": DATASET_ROOT / "cicids2017",
    "CIC-IDS2018": DATASET_ROOT / "cicids2018",
    "UNSW-NB15": DATASET_ROOT / "unsw_nb15",
}


# ============================================================
# PROCESS ONE FILE
# ============================================================

def investigate_file(file, dataset_name):

    print("\n" + "=" * 70)
    print(f"{dataset_name}")
    print(f"{file.name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    df = pd.read_csv(
        file,
        low_memory=False
    )

    total_rows = len(df)

    print(f"Total rows: {total_rows}")

    # --------------------------------------------------------
    # SHOW ORIGINAL COLUMNS
    # --------------------------------------------------------

    print("\nDetected columns:")

    for column in df.columns:
        print(f"  {repr(column)}")

    # --------------------------------------------------------
    # NORMALIZE COLUMN NAMES FOR DETECTION
    # --------------------------------------------------------
    #
    # Example:
    #
    # " Label" -> "label"
    # "Label " -> "label"
    # " ATTACK_CAT " -> "attack_cat"
    #
    # We DO NOT rename the actual dataframe columns.
    # We only use normalized names for detection.
    # --------------------------------------------------------

    normalized_columns = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    # --------------------------------------------------------
    # FIND LABEL COLUMN
    # --------------------------------------------------------

    label_column = None

    possible_label_columns = [
        "label",
        "attack_cat",
        "attack"
    ]

    for candidate in possible_label_columns:

        if candidate in normalized_columns:

            label_column = normalized_columns[
                candidate
            ]

            break

    print(
        f"\nLabel column: {label_column}"
    )

    # --------------------------------------------------------
    # DUPLICATE MASK
    # --------------------------------------------------------

    duplicate_mask = df.duplicated(
        keep="first"
    )

    duplicate_df = df[
        duplicate_mask
    ]

    unique_df = df[
        ~duplicate_mask
    ]

    duplicate_count = len(
        duplicate_df
    )

    unique_count = len(
        unique_df
    )

    duplicate_percentage = (
        duplicate_count / total_rows * 100
        if total_rows
        else 0
    )

    print(
        f"Duplicate rows: {duplicate_count}"
    )

    print(
        f"Unique rows: {unique_count}"
    )

    print(
        f"Duplicate percentage: "
        f"{duplicate_percentage:.4f}%"
    )

    # ========================================================
    # LABEL DISTRIBUTION
    # ========================================================

    original_labels = {}

    duplicate_labels = {}

    unique_labels = {}

    # --------------------------------------------------------
    # Original dataset labels
    # --------------------------------------------------------

    if label_column is not None:

        original_labels = (
            df[label_column]
            .value_counts(
                dropna=False
            )
            .to_dict()
        )

    # --------------------------------------------------------
    # Labels of duplicate rows
    # --------------------------------------------------------

    if (
        label_column is not None
        and duplicate_count > 0
    ):

        duplicate_labels = (
            duplicate_df[label_column]
            .value_counts(
                dropna=False
            )
            .to_dict()
        )

    # --------------------------------------------------------
    # Labels after keeping unique rows
    # --------------------------------------------------------

    if label_column is not None:

        unique_labels = (
            unique_df[label_column]
            .value_counts(
                dropna=False
            )
            .to_dict()
        )

    # --------------------------------------------------------
    # Convert dictionary keys safely to strings
    # --------------------------------------------------------

    original_labels = {
        str(k): int(v)
        for k, v in original_labels.items()
    }

    duplicate_labels = {
        str(k): int(v)
        for k, v in duplicate_labels.items()
    }

    unique_labels = {
        str(k): int(v)
        for k, v in unique_labels.items()
    }

    # ========================================================
    # RESULT
    # ========================================================

    result = {
        "dataset": dataset_name,
        "file": file.name,

        "rows_before": total_rows,

        "rows_after_removing_duplicates": unique_count,

        "duplicate_rows": duplicate_count,

        "duplicate_percentage": round(
            duplicate_percentage,
            4
        ),

        "label_column": label_column,

        "original_label_distribution":
            original_labels,

        "duplicate_label_distribution":
            duplicate_labels,

        "unique_label_distribution":
            unique_labels
    }

    # ========================================================
    # PRINT LABEL INFORMATION
    # ========================================================

    print("\nOriginal label distribution:")

    if original_labels:

        for label, count in (
            original_labels.items()
        ):

            print(
                f"  {label}: {count}"
            )

    else:

        print(
            "  No label column detected"
        )

    # --------------------------------------------------------

    print(
        "\nDuplicate rows by label:"
    )

    if duplicate_labels:

        for label, count in (
            duplicate_labels.items()
        ):

            print(
                f"  {label}: {count}"
            )

    else:

        print(
            "  No duplicate-label data"
        )

    # --------------------------------------------------------

    print(
        "\nUnique rows by label:"
    )

    if unique_labels:

        for label, count in (
            unique_labels.items()
        ):

            print(
                f"  {label}: {count}"
            )

    else:

        print(
            "  No unique-label data"
        )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("CyberSecAI")
    print("DUPLICATE INVESTIGATION")
    print("=" * 70)

    report = {
        "project": "CyberSecAI",
        "investigator_version": "1.1",
        "files": []
    }

    # --------------------------------------------------------
    # Datasets requiring duplicate investigation
    # --------------------------------------------------------

    selected_datasets = [
        "CIC-IDS2017",
        "UNSW-NB15"
    ]

    # ========================================================
    # PROCESS DATASETS
    # ========================================================

    for dataset_name in selected_datasets:

        print("\n")
        print("#" * 70)
        print(f"DATASET: {dataset_name}")
        print("#" * 70)

        dataset_path = DATASET_PATHS[
            dataset_name
        ]

        # ----------------------------------------------------
        # Check dataset directory
        # ----------------------------------------------------

        if not dataset_path.exists():

            print(
                f"❌ Dataset directory not found:"
            )

            print(dataset_path)

            continue

        # ----------------------------------------------------
        # Find CSV files
        # ----------------------------------------------------

        csv_files = sorted(
            [
                file
                for file in dataset_path.iterdir()
                if (
                    file.is_file()
                    and file.suffix.lower() == ".csv"
                    and not file.name.startswith("._")
                )
            ]
        )

        print(
            f"\nCSV files found: {len(csv_files)}"
        )

        # ----------------------------------------------------
        # Process each CSV
        # ----------------------------------------------------

        for file in csv_files:

            result = investigate_file(
                file,
                dataset_name
            )

            report["files"].append(
                result
            )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("DUPLICATE INVESTIGATION COMPLETE")
    print("=" * 70)

    print(
        "\nReport saved to:"
    )

    print(
        OUTPUT_FILE
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()