import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = Path(
    "/Volumes/Ganesh's SD/Cybersecurity-Project/datasets/raw"
)

MAPPING_FILE = (
    ROOT / "data" / "metadata" / "schema_label_mapping.json"
)

OUTPUT_FILE = (
    ROOT / "data" / "metadata" / "data_quality_validation.json"
)


# ============================================================
# DATASET LOCATIONS
# ============================================================

DATASET_PATHS = {
    "CIC-IDS2017": DATASET_ROOT / "cicids2017",
    "CIC-IDS2018": DATASET_ROOT / "cicids2018",
    "UNSW-NB15": DATASET_ROOT / "unsw_nb15",
}


# ============================================================
# VALIDATION SETTINGS
# ============================================================

# Read data in chunks so large CSV files do not need to fit
# completely into RAM.
CHUNK_SIZE = 100_000


# ============================================================
# LOAD MAPPING
# ============================================================

with open(MAPPING_FILE, "r", encoding="utf-8") as f:
    mapping = json.load(f)


# ============================================================
# HELPERS
# ============================================================

def clean_label(value):
    """Normalize label values."""

    label = str(value).strip().lower()

    label = label.replace("�", "-")
    label = label.replace("–", "-")
    label = label.replace("—", "-")

    label = " ".join(label.split())

    return label


def find_column(columns, name):
    """
    Find a column without being affected by
    leading/trailing spaces or case.
    """

    normalized = {
        str(col).strip().lower(): col
        for col in columns
    }

    return normalized.get(name.lower())


def find_label_columns(columns, dataset_name):
    """
    Find the relevant label columns.
    """

    if dataset_name == "UNSW-NB15":

        attack_cat = find_column(
            columns,
            "attack_cat"
        )

        label = find_column(
            columns,
            "label"
        )

        return attack_cat, label

    label = find_column(
        columns,
        "label"
    )

    return label, None


def is_numeric_series(series):
    """
    Check whether a pandas series can be interpreted
    as numeric data.
    """

    converted = pd.to_numeric(
        series,
        errors="coerce"
    )

    return converted.notna().all()


# ============================================================
# VALIDATE ONE FILE
# ============================================================

def validate_file(
    file,
    dataset_name,
    label_mapping
):

    print("\n" + "-" * 70)
    print(f"FILE: {file.name}")
    print("-" * 70)

    result = {
        "file": file.name,
        "status": "VALID",
        "rows": 0,
        "columns": 0,
        "missing_values": 0,
        "infinite_values": 0,
        "duplicate_rows": 0,
        "unknown_labels": [],
        "label_distribution": {},
        "issues": []
    }

    # --------------------------------------------------------
    # Read header
    # --------------------------------------------------------

    try:

        header = pd.read_csv(
            file,
            nrows=0
        )

    except Exception as e:

        result["status"] = "ERROR"
        result["issues"].append(
            f"Unable to read header: {e}"
        )

        return result

    columns = list(header.columns)

    result["columns"] = len(columns)

    print(f"Columns: {len(columns)}")

    # --------------------------------------------------------
    # Find labels
    # --------------------------------------------------------

    attack_cat_col, binary_label_col = find_label_columns(
        columns,
        dataset_name
    )

    if attack_cat_col is None:

        result["status"] = "CHECK_REQUIRED"

        result["issues"].append(
            "Label column not found"
        )

        print("❌ Label column not found")

        return result

    print(
        f"Primary label column: {attack_cat_col}"
    )

    if binary_label_col:

        print(
            f"Binary label column: {binary_label_col}"
        )

    # --------------------------------------------------------
    # Prepare mappings
    # --------------------------------------------------------

    normalized_mapping = {
        clean_label(key): value
        for key, value in label_mapping.items()
    }

    # --------------------------------------------------------
    # Chunk processing
    # --------------------------------------------------------

    total_missing = 0
    total_infinite = 0
    total_rows = 0

    label_counts = {}

    duplicate_hashes = set()
    duplicate_rows = 0

    binary_label_counts = {}

    unsw_label_mismatch = 0

    try:

        for chunk_number, chunk in enumerate(
            pd.read_csv(
                file,
                chunksize=CHUNK_SIZE,
                low_memory=False
            ),
            start=1
        ):

            print(
                f"Processing chunk {chunk_number}..."
            )

            rows = len(chunk)

            total_rows += rows

            # ------------------------------------------------
            # Missing values
            # ------------------------------------------------

            missing_count = int(
                chunk.isna().sum().sum()
            )

            total_missing += missing_count

            # ------------------------------------------------
            # Infinite values
            # ------------------------------------------------

            numeric_chunk = chunk.select_dtypes(
                include=[np.number]
            )

            if not numeric_chunk.empty:

                infinite_count = int(
                    np.isinf(
                        numeric_chunk.to_numpy()
                    ).sum()
                )

            else:

                infinite_count = 0

            total_infinite += infinite_count

            # ------------------------------------------------
            # Label distribution
            # ------------------------------------------------

            labels = (
                chunk[attack_cat_col]
                .dropna()
                .astype(str)
                .str.strip()
            )

            for label, count in labels.value_counts().items():

                label_counts[label] = (
                    label_counts.get(label, 0)
                    + int(count)
                )

            # ------------------------------------------------
            # UNSW label consistency
            # ------------------------------------------------

            if (
                dataset_name == "UNSW-NB15"
                and binary_label_col is not None
            ):

                attack_values = (
                    chunk[attack_cat_col]
                    .astype(str)
                    .str.strip()
                )

                binary_values = pd.to_numeric(
                    chunk[binary_label_col],
                    errors="coerce"
                )

                expected_binary = attack_values.map(
                    lambda x:
                    normalized_mapping.get(
                        clean_label(x),
                        {}
                    ).get(
                        "binary_label",
                        np.nan
                    )
                )

                mismatch = (
                    binary_values.notna()
                    & expected_binary.notna()
                    & (
                        binary_values.astype(float)
                        != expected_binary.astype(float)
                    )
                )

                unsw_label_mismatch += int(
                    mismatch.sum()
                )

                for value, count in (
                    binary_values
                    .value_counts(dropna=False)
                    .items()
                ):

                    key = str(value)

                    binary_label_counts[key] = (
                        binary_label_counts.get(key, 0)
                        + int(count)
                    )

            # ------------------------------------------------
            # Duplicate rows
            # ------------------------------------------------
            #
            # Hashing allows duplicate detection without
            # storing every complete row in memory.
            # ------------------------------------------------

            row_hashes = pd.util.hash_pandas_object(
                chunk,
                index=False
            )

            for row_hash in row_hashes:

                hash_value = int(row_hash)

                if hash_value in duplicate_hashes:

                    duplicate_rows += 1

                else:

                    duplicate_hashes.add(
                        hash_value
                    )

    except Exception as e:

        result["status"] = "ERROR"

        result["issues"].append(
            f"Error while processing file: {e}"
        )

        return result

    # --------------------------------------------------------
    # Store statistics
    # --------------------------------------------------------

    result["rows"] = total_rows

    result["missing_values"] = total_missing

    result["infinite_values"] = total_infinite

    result["duplicate_rows"] = duplicate_rows

    result["label_distribution"] = label_counts

    if binary_label_counts:

        result["binary_label_distribution"] = (
            binary_label_counts
        )

    result["unsw_label_mismatches"] = (
        unsw_label_mismatch
    )

    # --------------------------------------------------------
    # Unknown labels
    # --------------------------------------------------------

    unknown_labels = []

    for label in label_counts:

        if (
            clean_label(label)
            not in normalized_mapping
        ):

            unknown_labels.append(label)

    result["unknown_labels"] = unknown_labels

    # --------------------------------------------------------
    # Determine issues
    # --------------------------------------------------------

    if total_missing > 0:

        result["issues"].append(
            f"Missing values detected: {total_missing}"
        )

    if total_infinite > 0:

        result["issues"].append(
            f"Infinite values detected: {total_infinite}"
        )

    if duplicate_rows > 0:

        result["issues"].append(
            f"Duplicate rows detected: {duplicate_rows}"
        )

    if unknown_labels:

        result["issues"].append(
            "Unknown labels detected"
        )

    if unsw_label_mismatch > 0:

        result["issues"].append(
            "UNSW label/attack category mismatch detected"
        )

    if result["issues"]:

        result["status"] = "CHECK_REQUIRED"

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(f"Rows: {total_rows}")

    print(
        f"Missing values: {total_missing}"
    )

    print(
        f"Infinite values: {total_infinite}"
    )

    print(
        f"Duplicate rows: {duplicate_rows}"
    )

    print("\nLabel distribution:")

    for label, count in label_counts.items():

        print(
            f"   {label}: {count}"
        )

    if dataset_name == "UNSW-NB15":

        print(
            f"\nUNSW label mismatches: "
            f"{unsw_label_mismatch}"
        )

    if unknown_labels:

        print("\n❌ Unknown labels:")

        for label in unknown_labels:

            print(
                f"   → {label}"
            )

    if result["status"] == "VALID":

        print("\n✅ DATA QUALITY VALID")

    else:

        print("\n⚠️ CHECK REQUIRED")

        for issue in result["issues"]:

            print(
                f"   → {issue}"
            )

    return result


# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_dataset(
    dataset_name,
    config
):

    print("\n")
    print("=" * 70)
    print(dataset_name)
    print("=" * 70)

    dataset_path = DATASET_PATHS[
        dataset_name
    ]

    print(
        f"Dataset path:\n{dataset_path}"
    )

    if not dataset_path.exists():

        return {
            "status": "DIRECTORY_NOT_FOUND",
            "path": str(dataset_path),
            "files": []
        }

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
        f"CSV files found: {len(csv_files)}"
    )

    label_mapping = config[
        "label_mapping"
    ]

    file_results = []

    for file in csv_files:

        result = validate_file(
            file,
            dataset_name,
            label_mapping
        )

        file_results.append(result)

    dataset_status = (
        "VALID"
        if file_results
        and all(
            result["status"] == "VALID"
            for result in file_results
        )
        else "CHECK_REQUIRED"
    )

    return {
        "status": dataset_status,
        "path": str(dataset_path),
        "file_count": len(file_results),
        "files": file_results
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("CyberSecAI")
    print("DATA QUALITY VALIDATION")
    print("=" * 70)

    final_result = {
        "project": "CyberSecAI",
        "validator_version": "1.0",
        "dataset_root": str(DATASET_ROOT),
        "datasets": {}
    }

    # --------------------------------------------------------
    # Validate all datasets
    # --------------------------------------------------------

    for dataset_name, config in mapping[
        "datasets"
    ].items():

        result = validate_dataset(
            dataset_name,
            config
        )

        final_result[
            "datasets"
        ][dataset_name] = result

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

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
            final_result,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n\n")
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    all_valid = True

    for dataset_name, result in (
        final_result["datasets"].items()
    ):

        if result["status"] == "VALID":

            print(
                f"✅ {dataset_name}: VALID"
            )

        else:

            print(
                f"⚠️ {dataset_name}: CHECK REQUIRED"
            )

            all_valid = False

    print("\n")

    if all_valid:

        print(
            "🎉 DATA QUALITY VALIDATION PASSED"
        )

    else:

        print(
            "⚠️ DATA QUALITY ISSUES REQUIRE REVIEW"
        )

    print("\nValidation report:")
    print(OUTPUT_FILE)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()