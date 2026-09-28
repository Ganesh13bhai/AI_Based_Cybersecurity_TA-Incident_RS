from pathlib import Path
import pandas as pd
import json
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = Path(
    "/Volumes/Ganesh's SD/Cybersecurity-Project/datasets/raw"
)

OUTPUT_FILE = ROOT / "data" / "metadata" / "cross_file_duplicate_analysis.json"

DATASETS = {
    "CIC-IDS2017": DATASET_ROOT / "cicids2017",
    "CIC-IDS2018": DATASET_ROOT / "cicids2018",
    "UNSW-NB15": DATASET_ROOT / "unsw_nb15",
}


# ============================================================
# HELPERS
# ============================================================

def detect_label_column(columns):
    """
    Detect label column using normalized column names.
    """
    normalized = {
        str(col).strip().lower(): col
        for col in columns
    }

    for candidate in ["attack_cat", "label", "attack"]:
        if candidate in normalized:
            return normalized[candidate]

    return None


def clean_label(value):
    if pd.isna(value):
        return "UNKNOWN"

    label = str(value).strip()

    label = label.replace("�", "-")
    label = label.replace("–", "-")
    label = label.replace("—", "-")

    label = " ".join(label.split())

    return label


def get_csv_files(dataset_path):
    """
    Ignore macOS metadata files such as ._filename.csv
    """
    return sorted(
        file
        for file in dataset_path.glob("*.csv")
        if not file.name.startswith("._")
    )


# ============================================================
# ANALYZE ONE DATASET
# ============================================================

def analyze_dataset(dataset_name, dataset_path):

    print()
    print("=" * 80)
    print(f"DATASET: {dataset_name}")
    print("=" * 80)

    files = get_csv_files(dataset_path)

    # --------------------------------------------------------
    # First pass:
    # Read each file and create row hashes
    # --------------------------------------------------------

    file_hashes = {}
    file_labels = {}
    file_row_counts = {}

    global_hash_locations = defaultdict(list)

    for file_path in files:

        print()
        print(f"Reading: {file_path.name}")

        df = pd.read_csv(
            file_path,
            low_memory=False
        )

        row_count = len(df)

        label_column = detect_label_column(df.columns)

        if label_column is not None:
            labels = df[label_column].map(clean_label)
        else:
            labels = pd.Series(
                ["UNKNOWN"] * row_count,
                index=df.index
            )

        # Complete-row hash
        hashes = pd.util.hash_pandas_object(
            df,
            index=False
        )

        file_hashes[file_path.name] = hashes
        file_labels[file_path.name] = labels
        file_row_counts[file_path.name] = row_count

        # Store where each hash occurs
        for row_index, row_hash in enumerate(hashes):

            global_hash_locations[int(row_hash)].append(
                {
                    "file": file_path.name,
                    "row_index": row_index,
                    "label": labels.iloc[row_index]
                }
            )

        print(f"Rows: {row_count:,}")
        print(f"Label column: {label_column}")

    # --------------------------------------------------------
    # Cross-file duplicate groups
    # --------------------------------------------------------

    cross_file_groups = []

    for row_hash, locations in global_hash_locations.items():

        files_in_group = sorted(
            set(item["file"] for item in locations)
        )

        # Only interested in records occurring
        # in MORE THAN ONE FILE.
        if len(files_in_group) > 1:

            cross_file_groups.append(
                {
                    "row_hash": row_hash,
                    "file_count": len(files_in_group),
                    "files": files_in_group,
                    "occurrences": locations,
                    "total_repetitions": len(locations),
                }
            )

    # Largest groups first
    cross_file_groups.sort(
        key=lambda x: x["total_repetitions"],
        reverse=True
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_rows = sum(file_row_counts.values())

    total_cross_file_rows = sum(
        group["total_repetitions"]
        for group in cross_file_groups
    )

    # Count unique rows participating in cross-file duplication
    cross_file_unique_groups = len(cross_file_groups)

    # --------------------------------------------------------
    # File-pair overlap
    # --------------------------------------------------------

    pair_overlap = defaultdict(int)

    for group in cross_file_groups:

        files_in_group = group["files"]

        for i in range(len(files_in_group)):

            for j in range(i + 1, len(files_in_group)):

                pair = (
                    files_in_group[i],
                    files_in_group[j]
                )

                pair_overlap[pair] += 1

    pair_overlap_list = []

    for pair, count in sorted(
        pair_overlap.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        pair_overlap_list.append(
            {
                "file_a": pair[0],
                "file_b": pair[1],
                "shared_duplicate_groups": count
            }
        )

    # --------------------------------------------------------
    # Label consistency
    # --------------------------------------------------------

    label_conflicts = []

    for group in cross_file_groups:

        labels = sorted(
            set(
                occurrence["label"]
                for occurrence in group["occurrences"]
            )
        )

        if len(labels) > 1:

            label_conflicts.append(
                {
                    "row_hash": group["row_hash"],
                    "files": group["files"],
                    "labels": labels,
                    "occurrences": group["occurrences"]
                }
            )

    # --------------------------------------------------------
    # Top duplicate groups
    # --------------------------------------------------------

    top_groups = cross_file_groups[:50]

    result = {
        "dataset": dataset_name,

        "files_analyzed": len(files),

        "file_row_counts": file_row_counts,

        "total_rows": total_rows,

        "cross_file_duplicate_groups": cross_file_unique_groups,

        "rows_in_cross_file_duplicate_groups": total_cross_file_rows,

        "cross_file_duplicate_percentage": (
            round(
                total_cross_file_rows / total_rows * 100,
                4
            )
            if total_rows
            else 0
        ),

        "file_pair_overlap": pair_overlap_list,

        "label_conflict_group_count": len(label_conflicts),

        "label_conflicts": label_conflicts[:100],

        "top_cross_file_duplicate_groups": top_groups,
    }

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("CROSS-FILE DUPLICATE / DATA LEAKAGE ANALYSIS")
    print("=" * 80)

    all_results = {}

    for dataset_name, dataset_path in DATASETS.items():

        if not dataset_path.exists():

            print(
                f"\nWARNING: Dataset path does not exist: "
                f"{dataset_path}"
            )

            continue

        try:

            result = analyze_dataset(
                dataset_name,
                dataset_path
            )

            all_results[dataset_name] = result

        except Exception as error:

            print(
                f"\nERROR processing {dataset_name}: "
                f"{error}"
            )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report = {
        "project": "CyberSecAI",

        "purpose": (
            "Identify identical complete rows occurring "
            "across different CSV files within each dataset."
        ),

        "method": (
            "Complete-row hashing using "
            "pandas hash_pandas_object(index=False)."
        ),

        "raw_data_modified": False,

        "datasets": all_results
    }

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

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    for dataset_name, result in all_results.items():

        print()
        print(f"Dataset: {dataset_name}")
        print(
            f"Total rows: "
            f"{result['total_rows']:,}"
        )
        print(
            f"Cross-file duplicate groups: "
            f"{result['cross_file_duplicate_groups']:,}"
        )
        print(
            f"Rows involved in cross-file duplicates: "
            f"{result['rows_in_cross_file_duplicate_groups']:,}"
        )
        print(
            f"Cross-file duplicate percentage: "
            f"{result['cross_file_duplicate_percentage']}%"
        )
        print(
            f"Label-conflict groups: "
            f"{result['label_conflict_group_count']:,}"
        )

    print()
    print("=" * 80)
    print(f"Report saved to:")
    print(OUTPUT_FILE)
    print("=" * 80)


if __name__ == "__main__":
    main()