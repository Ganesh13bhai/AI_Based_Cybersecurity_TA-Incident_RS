"""
CyberSecAI - Duplicate Group Analysis

Purpose:
    Analyze how exact duplicate rows are distributed inside each dataset file.

This script DOES NOT modify raw datasets.

It analyzes:
    - duplicate groups
    - repetition frequency
    - labels inside duplicate groups
    - number of unique rows
    - maximum repetition
    - highly repeated rows
    - duplicate contribution by label

Output:
    data/metadata/duplicate_group_analysis.json
"""

from pathlib import Path
from collections import Counter
import json
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = Path(
    "/Volumes/Ganesh's SD/Cybersecurity-Project/datasets/raw"
)

OUTPUT_FILE = ROOT / "data" / "metadata" / "duplicate_group_analysis.json"


DATASET_PATHS = {
    "CIC-IDS2017": DATASET_ROOT / "cicids2017",
    "UNSW-NB15": DATASET_ROOT / "unsw_nb15",
}


# ============================================================
# CONFIGURATION
# ============================================================

CHUNK_SIZE = 100_000

# Number of most repeated duplicate groups to report
TOP_DUPLICATE_GROUPS = 20


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_label(value):
    """
    Normalize labels for reporting.
    """
    if pd.isna(value):
        return "UNKNOWN"

    label = str(value).strip()

    label = label.replace("�", "-")
    label = label.replace("–", "-")
    label = label.replace("—", "-")

    label = " ".join(label.split())

    return label


def find_label_column(columns):
    """
    Detect label column regardless of leading/trailing spaces
    or capitalization.
    """

    normalized_columns = {
        str(column).strip().lower(): column
        for column in columns
    }

    # Prefer attack_cat for UNSW family analysis if available
    candidates = [
        "attack_cat",
        "label",
        "attack",
    ]

    for candidate in candidates:
        if candidate in normalized_columns:
            return normalized_columns[candidate]

    return None


def make_row_hashes(df):
    """
    Generate stable hashes for complete rows.

    Every column is included, including the label.
    Therefore this represents EXACT duplicate rows.
    """

    return pd.util.hash_pandas_object(
        df,
        index=False
    )


# ============================================================
# ANALYZE ONE FILE
# ============================================================

def analyze_file(dataset_name, file_path):

    print()
    print("=" * 80)
    print(f"Dataset : {dataset_name}")
    print(f"File    : {file_path.name}")
    print("=" * 80)

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    try:
        df = pd.read_csv(
            file_path,
            low_memory=False
        )
    except Exception as exc:
        print(f"ERROR reading file: {exc}")
        return {
            "dataset": dataset_name,
            "file": file_path.name,
            "status": "ERROR",
            "error": str(exc),
        }

    rows_before = len(df)

    print(f"Rows: {rows_before:,}")
    print()

    # --------------------------------------------------------
    # Detect label
    # --------------------------------------------------------

    label_column = find_label_column(df.columns)

    print("Detected label column:", repr(label_column))

    if label_column is not None:
        print("Original label distribution:")

        original_labels = (
            df[label_column]
            .map(clean_label)
            .value_counts()
            .to_dict()
        )

        for label, count in original_labels.items():
            print(f"  {label}: {count:,}")
    else:
        original_labels = {}

    # --------------------------------------------------------
    # Calculate exact row hashes
    # --------------------------------------------------------

    print()
    print("Calculating row hashes...")

    row_hashes = make_row_hashes(df)

    hash_counts = row_hashes.value_counts()

    # Number of unique rows
    unique_rows = len(hash_counts)

    duplicate_rows = rows_before - unique_rows

    duplicate_percentage = (
        duplicate_rows / rows_before * 100
        if rows_before
        else 0
    )

    print(f"Unique rows       : {unique_rows:,}")
    print(f"Duplicate rows    : {duplicate_rows:,}")
    print(f"Duplicate percent : {duplicate_percentage:.4f}%")

    # --------------------------------------------------------
    # Duplicate groups
    # --------------------------------------------------------

    duplicate_group_counts = hash_counts[
        hash_counts > 1
    ]

    duplicate_group_count = len(duplicate_group_counts)

    print(
        f"Duplicate groups  : "
        f"{duplicate_group_count:,}"
    )

    # --------------------------------------------------------
    # Repetition statistics
    # --------------------------------------------------------

    if duplicate_group_count > 0:

        repetitions = duplicate_group_counts.values

        max_repetition = int(repetitions.max())

        average_repetition = float(repetitions.mean())

        median_repetition = float(
            pd.Series(repetitions).median()
        )

    else:

        max_repetition = 1
        average_repetition = 1.0
        median_repetition = 1.0

    print(
        f"Maximum repetition: "
        f"{max_repetition}"
    )

    print(
        f"Average repetition: "
        f"{average_repetition:.2f}"
    )

    print(
        f"Median repetition : "
        f"{median_repetition:.2f}"
    )

    # --------------------------------------------------------
    # Frequency distribution
    # --------------------------------------------------------

    frequency_distribution = Counter(
        int(value)
        for value in hash_counts.values
    )

    frequency_distribution = {
        str(k): int(v)
        for k, v in sorted(
            frequency_distribution.items()
        )
    }

    # --------------------------------------------------------
    # Top duplicate groups
    # --------------------------------------------------------

    top_duplicate_groups = []

    if duplicate_group_count > 0:

        top_groups = (
            duplicate_group_counts
            .sort_values(ascending=False)
            .head(TOP_DUPLICATE_GROUPS)
        )

        for row_hash, repetition_count in top_groups.items():

            matching_indices = (
                row_hashes[
                    row_hashes == row_hash
                ].index
                .tolist()
            )

            group_info = {
                "row_hash": str(row_hash),
                "repetition_count": int(
                    repetition_count
                ),
                "example_row_index": int(
                    matching_indices[0]
                ),
            }

            # ------------------------------------------------
            # Label information
            # ------------------------------------------------

            if label_column is not None:

                labels = (
                    df.loc[
                        matching_indices,
                        label_column
                    ]
                    .map(clean_label)
                    .value_counts()
                    .to_dict()
                )

                group_info["labels"] = {
                    str(label): int(count)
                    for label, count in labels.items()
                }

            top_duplicate_groups.append(
                group_info
            )

    # --------------------------------------------------------
    # Duplicate rows by label
    # --------------------------------------------------------

    duplicate_label_distribution = {}

    if label_column is not None:

        duplicate_mask = row_hashes.duplicated(
            keep="first"
        )

        duplicate_labels = (
            df.loc[
                duplicate_mask,
                label_column
            ]
            .map(clean_label)
            .value_counts()
            .to_dict()
        )

        duplicate_label_distribution = {
            str(label): int(count)
            for label, count in
            duplicate_labels.items()
        }

        print()
        print("Duplicate rows by label:")

        for label, count in duplicate_label_distribution.items():
            print(f"  {label}: {count:,}")

    # --------------------------------------------------------
    # Unique-row label distribution
    # --------------------------------------------------------

    unique_label_distribution = {}

    if label_column is not None:

        unique_mask = ~row_hashes.duplicated(
            keep="first"
        )

        unique_labels = (
            df.loc[
                unique_mask,
                label_column
            ]
            .map(clean_label)
            .value_counts()
            .to_dict()
        )

        unique_label_distribution = {
            str(label): int(count)
            for label, count in
            unique_labels.items()
        }

        print()
        print("Unique rows by label:")

        for label, count in unique_label_distribution.items():
            print(f"  {label}: {count:,}")

    # --------------------------------------------------------
    # Label-specific duplicate percentages
    # --------------------------------------------------------

    label_duplicate_percentages = {}

    if label_column is not None:

        for label, original_count in original_labels.items():

            duplicate_count = duplicate_label_distribution.get(
                label,
                0
            )

            percentage = (
                duplicate_count / original_count * 100
                if original_count
                else 0
            )

            label_duplicate_percentages[label] = round(
                percentage,
                4
            )

    # --------------------------------------------------------
    # Build result
    # --------------------------------------------------------

    result = {
        "dataset": dataset_name,
        "file": file_path.name,
        "status": "ANALYZED",

        "rows_before": int(rows_before),

        "unique_rows": int(unique_rows),

        "duplicate_rows": int(duplicate_rows),

        "duplicate_percentage": round(
            duplicate_percentage,
            4
        ),

        "duplicate_group_count": int(
            duplicate_group_count
        ),

        "maximum_repetition": int(
            max_repetition
        ),

        "average_repetition": round(
            average_repetition,
            4
        ),

        "median_repetition": round(
            median_repetition,
            4
        ),

        "label_column": label_column,

        "original_label_distribution": {
            str(label): int(count)
            for label, count in
            original_labels.items()
        },

        "duplicate_label_distribution": {
            str(label): int(count)
            for label, count in
            duplicate_label_distribution.items()
        },

        "unique_label_distribution": {
            str(label): int(count)
            for label, count in
            unique_label_distribution.items()
        },

        "label_duplicate_percentages": (
            label_duplicate_percentages
        ),

        "repetition_frequency_distribution": (
            frequency_distribution
        ),

        "top_duplicate_groups": (
            top_duplicate_groups
        ),
    }

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("CyberSecAI - Duplicate Group Analysis")
    print("=" * 80)
    print()

    results = []

    for dataset_name, dataset_path in DATASET_PATHS.items():

        if not dataset_path.exists():

            print(
                f"WARNING: Dataset path does not exist:"
                f"\n{dataset_path}"
            )

            continue

        csv_files = sorted(
            file
            for file in dataset_path.glob("*.csv")
            if not file.name.startswith("._")
        )

        print(
            f"\n{dataset_name}: "
            f"{len(csv_files)} CSV files found"
        )

        for file_path in csv_files:

            result = analyze_file(
                dataset_name,
                file_path
            )

            results.append(result)

    # ========================================================
    # SUMMARY
    # ========================================================

    analyzed_results = [
        result
        for result in results
        if result.get("status") == "ANALYZED"
    ]

    total_rows = sum(
        result["rows_before"]
        for result in analyzed_results
    )

    total_unique_rows = sum(
        result["unique_rows"]
        for result in analyzed_results
    )

    total_duplicate_rows = sum(
        result["duplicate_rows"]
        for result in analyzed_results
    )

    overall_duplicate_percentage = (
        total_duplicate_rows / total_rows * 100
        if total_rows
        else 0
    )

    summary = {
        "files_analyzed": len(analyzed_results),

        "total_rows": int(
            total_rows
        ),

        "total_unique_rows": int(
            total_unique_rows
        ),

        "total_duplicate_rows": int(
            total_duplicate_rows
        ),

        "overall_duplicate_percentage": round(
            overall_duplicate_percentage,
            4
        ),
    }

    # ========================================================
    # FINAL REPORT
    # ========================================================

    report = {
        "project": "CyberSecAI",

        "investigator_version": "1.0",

        "description": (
            "Analysis of exact duplicate row groups "
            "without modifying raw datasets."
        ),

        "method": (
            "Complete-row hashing using all columns. "
            "Rows with identical values across all "
            "columns are considered exact duplicates."
        ),

        "summary": summary,

        "files": results,
    }

    # ========================================================
    # SAVE JSON
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 80)
    print("DUPLICATE GROUP ANALYSIS COMPLETE")
    print("=" * 80)

    print()
    print(
        f"Files analyzed          : "
        f"{len(analyzed_results):,}"
    )

    print(
        f"Total rows              : "
        f"{total_rows:,}"
    )

    print(
        f"Total unique rows       : "
        f"{total_unique_rows:,}"
    )

    print(
        f"Total duplicate rows    : "
        f"{total_duplicate_rows:,}"
    )

    print(
        f"Overall duplicate rate  : "
        f"{overall_duplicate_percentage:.4f}%"
    )

    print()
    print("Report saved to:")
    print(OUTPUT_FILE)

    print()
    print(
        "IMPORTANT: Raw datasets were NOT modified."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()