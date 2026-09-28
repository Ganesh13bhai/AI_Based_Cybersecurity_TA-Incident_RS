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

OUTPUT_FILE = (
    ROOT / "data" / "metadata" / "data_quality_investigation.json"
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
# SETTINGS
# ============================================================

CHUNK_SIZE = 100_000


# ============================================================
# INVESTIGATE ONE FILE
# ============================================================

def investigate_file(file, dataset_name):

    print("\n" + "-" * 70)
    print(f"FILE: {file.name}")
    print("-" * 70)

    result = {
        "file": file.name,
        "dataset": dataset_name,
        "rows": 0,
        "columns": 0,
        "missing_by_column": {},
        "infinite_by_column": {},
        "duplicate_rows": 0,
        "duplicate_percentage": 0.0,
        "issues": []
    }

    # --------------------------------------------------------
    # Read header
    # --------------------------------------------------------

    header = pd.read_csv(
        file,
        nrows=0
    )

    columns = list(header.columns)

    result["columns"] = len(columns)

    print(f"Columns: {len(columns)}")

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    missing_counts = {
        column: 0
        for column in columns
    }

    infinite_counts = {
        column: 0
        for column in columns
    }

    total_rows = 0

    duplicate_hashes = set()
    duplicate_rows = 0

    # --------------------------------------------------------
    # Process chunks
    # --------------------------------------------------------

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

        total_rows += len(chunk)

        # ----------------------------------------------------
        # Missing values by column
        # ----------------------------------------------------

        chunk_missing = chunk.isna().sum()

        for column, count in chunk_missing.items():

            missing_counts[column] += int(count)

        # ----------------------------------------------------
        # Infinite values by numeric column
        # ----------------------------------------------------

        numeric_columns = chunk.select_dtypes(
            include=[np.number]
        ).columns

        for column in numeric_columns:

            values = pd.to_numeric(
                chunk[column],
                errors="coerce"
            )

            infinite_count = int(
                np.isinf(values.to_numpy()).sum()
            )

            infinite_counts[column] += infinite_count

        # ----------------------------------------------------
        # Exact duplicate rows
        # ----------------------------------------------------

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

    # ========================================================
    # RESULTS
    # ========================================================

    result["rows"] = total_rows

    result["duplicate_rows"] = duplicate_rows

    if total_rows > 0:

        result["duplicate_percentage"] = round(
            (duplicate_rows / total_rows) * 100,
            4
        )

    # --------------------------------------------------------
    # Keep only problematic columns
    # --------------------------------------------------------

    result["missing_by_column"] = {
        column: count
        for column, count in missing_counts.items()
        if count > 0
    }

    result["infinite_by_column"] = {
        column: count
        for column, count in infinite_counts.items()
        if count > 0
    }

    # --------------------------------------------------------
    # Print missing values
    # --------------------------------------------------------

    print("\nMissing values by column:")

    if result["missing_by_column"]:

        for column, count in (
            result["missing_by_column"].items()
        ):

            print(
                f"   {column}: {count}"
            )

    else:

        print("   None")

    # --------------------------------------------------------
    # Print infinite values
    # --------------------------------------------------------

    print("\nInfinite values by column:")

    if result["infinite_by_column"]:

        for column, count in (
            result["infinite_by_column"].items()
        ):

            print(
                f"   {column}: {count}"
            )

    else:

        print("   None")

    # --------------------------------------------------------
    # Duplicate information
    # --------------------------------------------------------

    print(
        f"\nRows: {total_rows}"
    )

    print(
        f"Exact duplicate rows: {duplicate_rows}"
    )

    print(
        f"Duplicate percentage: "
        f"{result['duplicate_percentage']}%"
    )

    # --------------------------------------------------------
    # Issues
    # --------------------------------------------------------

    if result["missing_by_column"]:

        result["issues"].append(
            "Missing values present"
        )

    if result["infinite_by_column"]:

        result["issues"].append(
            "Infinite values present"
        )

    if duplicate_rows > 0:

        result["issues"].append(
            "Duplicate rows present"
        )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("CyberSecAI")
    print("DATA QUALITY INVESTIGATION")
    print("=" * 70)

    final_result = {
        "project": "CyberSecAI",
        "investigator_version": "1.0",
        "datasets": {}
    }

    # ========================================================
    # DATASETS
    # ========================================================

    for dataset_name, dataset_path in (
        DATASET_PATHS.items()
    ):

        print("\n")
        print("=" * 70)
        print(dataset_name)
        print("=" * 70)

        if not dataset_path.exists():

            print(
                "❌ Dataset directory not found"
            )

            final_result["datasets"][
                dataset_name
            ] = {
                "status": "DIRECTORY_NOT_FOUND"
            }

            continue

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

        file_results = []

        for file in csv_files:

            result = investigate_file(
                file,
                dataset_name
            )

            file_results.append(result)

        final_result["datasets"][
            dataset_name
        ] = {
            "file_count": len(file_results),
            "files": file_results
        }

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
            final_result,
            f,
            indent=2
        )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n")
    print("=" * 70)
    print("INVESTIGATION COMPLETE")
    print("=" * 70)

    print(
        "\nReport saved to:"
    )

    print(OUTPUT_FILE)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()