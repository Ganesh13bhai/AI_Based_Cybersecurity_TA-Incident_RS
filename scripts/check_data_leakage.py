import json
import glob
import os

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PROCESSED_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

FEATURE_MAPPING_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "metadata",
    "feature_mapping.json"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "metadata",
    "leakage_report.json"
)


# ============================================================
# LABEL / TARGET COLUMNS
# ============================================================

# These columns contain target information and MUST NOT be
# used as model input features.

TARGET_COLUMNS = {
    "label",
    "attack_cat",
    "canonical_label",
    "binary_label"
}


# ============================================================
# SUSPICIOUS COLUMN PATTERNS
# ============================================================

# These names do not automatically mean leakage.
# They are flagged for manual investigation.

SUSPICIOUS_PATTERNS = {
    "attack",
    "target",
    "class",
    "prediction",
    "predicted",
    "ground_truth"
}


# ============================================================
# SETTINGS
# ============================================================

# Number of rows processed at a time during duplicate checks.
# This avoids loading multi-million-row CSVs completely.
CHUNK_SIZE = 100_000


# ============================================================
# DATASET IDENTIFICATION
# ============================================================

def get_dataset_name(path):

    path_lower = path.lower()

    if "cicids2017" in path_lower:
        return "CIC-IDS2017"

    if "cicids2018" in path_lower:
        return "CIC-IDS2018"

    if "unswnb15" in path_lower:
        return "UNSW-NB15"

    return "UNKNOWN"


# ============================================================
# DUPLICATE FEATURE ROW CHECK
# ============================================================

def count_duplicate_feature_rows(
    file,
    feature_columns
):
    """
    Count duplicate feature rows without loading the entire
    CSV into memory at once.

    Important:
    Duplicate rows are NOT automatically considered leakage.
    They are reported for further analysis.
    """

    if not feature_columns:
        return 0

    duplicate_count = 0

    # --------------------------------------------------------
    # We use a hash set of feature rows.
    # This allows duplicate detection across chunks.
    # --------------------------------------------------------

    seen_hashes = set()

    try:

        for chunk in pd.read_csv(
            file,
            usecols=feature_columns,
            chunksize=CHUNK_SIZE
        ):

            # Create a stable hash for every feature row
            row_hashes = pd.util.hash_pandas_object(
                chunk,
                index=False
            )

            for row_hash in row_hashes:

                row_hash = int(row_hash)

                if row_hash in seen_hashes:

                    duplicate_count += 1

                else:

                    seen_hashes.add(row_hash)

    except Exception as e:

        print(
            f"Duplicate check failed: {e}"
        )

        return None

    return duplicate_count


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CyberSecAI - Data Leakage Check")
    print("=" * 70)

    # --------------------------------------------------------
    # Find processed CSV files
    # --------------------------------------------------------

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

    # Ignore macOS AppleDouble files such as ._filename.csv
    files = [
        f
        for f in files
        if not os.path.basename(f).startswith("._")
    ]

    if not files:

        raise FileNotFoundError(
            f"No processed CSV files found in:\n"
            f"{PROCESSED_DIR}"
        )

    print(
        f"\nProcessed CSV files found: {len(files)}"
    )

    # --------------------------------------------------------
    # Load feature mapping
    # --------------------------------------------------------

    with open(
        FEATURE_MAPPING_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        feature_mapping = json.load(f)

    canonical_features = set(
        feature_mapping.keys()
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    results = []

    total_files = 0
    total_target_columns = 0
    total_suspicious_columns = 0
    total_duplicate_rows = 0

    # ========================================================
    # PROCESS EACH FILE
    # ========================================================

    for file in files:

        dataset = get_dataset_name(file)

        if dataset == "UNKNOWN":

            print(
                f"\nSkipping unknown dataset:"
                f" {os.path.basename(file)}"
            )

            continue

        total_files += 1

        print("\n" + "-" * 70)
        print(dataset)
        print(os.path.basename(file))
        print("-" * 70)

        # ----------------------------------------------------
        # Read only CSV header
        # ----------------------------------------------------

        try:

            columns = pd.read_csv(
                file,
                nrows=0
            ).columns.tolist()

        except Exception as e:

            print(
                f"ERROR reading header: {e}"
            )

            results.append(
                {
                    "dataset": dataset,
                    "file": os.path.basename(file),
                    "status": "ERROR",
                    "error": str(e)
                }
            )

            continue

        # ----------------------------------------------------
        # Normalize column names
        # ----------------------------------------------------

        normalized_columns = {
            str(column).strip().lower(): column
            for column in columns
        }

        # ====================================================
        # TARGET / LABEL COLUMNS
        # ====================================================

        target_columns = []

        for normalized, original in (
            normalized_columns.items()
        ):

            if normalized in TARGET_COLUMNS:

                target_columns.append(
                    original
                )

        total_target_columns += len(
            target_columns
        )

        print(
            "Target/label columns:"
        )

        if target_columns:

            for column in target_columns:

                print(
                    f"  - {column}"
                )

        else:

            print("  NONE")

        # ====================================================
        # SUSPICIOUS COLUMNS
        # ====================================================

        suspicious_columns = []

        for column in columns:

            normalized = (
                str(column)
                .strip()
                .lower()
            )

            # Target columns are already expected
            # and therefore not flagged here.
            if normalized in TARGET_COLUMNS:
                continue

            for pattern in SUSPICIOUS_PATTERNS:

                if pattern in normalized:

                    suspicious_columns.append(
                        {
                            "column": column,
                            "reason": (
                                f"contains '{pattern}'"
                            )
                        }
                    )

                    break

        total_suspicious_columns += len(
            suspicious_columns
        )

        if suspicious_columns:

            print(
                "Suspicious columns:"
            )

            for item in suspicious_columns:

                print(
                    f"  - {item['column']} "
                    f"({item['reason']})"
                )

        else:

            print(
                "Suspicious columns: NONE"
            )

        # ====================================================
        # FEATURE MAPPING COVERAGE
        # ====================================================

        canonical_present = []

        for feature in canonical_features:

            mapping = feature_mapping[
                feature
            ]

            # ------------------------------------------------
            # Case 1:
            # Mapping is a dictionary.
            # ------------------------------------------------

            if isinstance(
                mapping,
                dict
            ):

                dataset_mapping = mapping.get(
                    dataset,
                    {}
                )

                if dataset_mapping:

                    canonical_present.append(
                        feature
                    )

            # ------------------------------------------------
            # Case 2:
            # Mapping is a string.
            # ------------------------------------------------

            elif isinstance(
                mapping,
                str
            ):

                if mapping.strip():

                    canonical_present.append(
                        feature
                    )

        print(
            "Canonical features available: "
            f"{len(canonical_present)}"
        )

        # ====================================================
        # FEATURE COLUMNS FOR MODEL
        # ====================================================

        feature_columns = []

        for column in columns:

            normalized = (
                str(column)
                .strip()
                .lower()
            )

            if normalized not in TARGET_COLUMNS:

                feature_columns.append(
                    column
                )

        print(
            "Potential model feature columns: "
            f"{len(feature_columns)}"
        )

        # ====================================================
        # DUPLICATE FEATURE ROW CHECK
        # ====================================================

        print(
            "Checking duplicate feature rows "
            f"in chunks of {CHUNK_SIZE:,}..."
        )

        duplicate_rows = (
            count_duplicate_feature_rows(
                file,
                feature_columns
            )
        )

        if duplicate_rows is not None:

            total_duplicate_rows += (
                duplicate_rows
            )

            print(
                "Duplicate feature rows: "
                f"{duplicate_rows:,}"
            )

        else:

            print(
                "Duplicate feature rows: "
                "CHECK FAILED"
            )

        # ====================================================
        # FILE STATUS
        # ====================================================

        # Target columns are EXPECTED.
        #
        # Suspicious columns require investigation.
        #
        # Duplicate rows are NOT automatically leakage.

        if suspicious_columns:

            status = "REVIEW"

        else:

            status = "PASS"

        results.append(
            {
                "dataset": dataset,
                "file": os.path.basename(file),
                "columns": len(columns),
                "target_columns": target_columns,
                "suspicious_columns": (
                    suspicious_columns
                ),
                "canonical_features_available": len(
                    canonical_present
                ),
                "model_feature_columns": len(
                    feature_columns
                ),
                "duplicate_feature_rows": (
                    duplicate_rows
                ),
                "status": status
            }
        )

    # ========================================================
    # OVERALL STATUS
    # ========================================================

    suspicious_files = [
        result
        for result in results
        if result.get(
            "suspicious_columns"
        )
    ]

    error_files = [
        result
        for result in results
        if result.get("status") == "ERROR"
    ]

    if error_files:

        overall_status = "ERROR"

    elif suspicious_files:

        overall_status = "REVIEW_REQUIRED"

    else:

        overall_status = "PASS"

    # ========================================================
    # REPORT
    # ========================================================

    report = {

        "version": "1.0",

        "description": (
            "Structural data leakage audit for "
            "CyberSecAI processed datasets."
        ),

        "target_columns": sorted(
            TARGET_COLUMNS
        ),

        "suspicious_patterns": sorted(
            SUSPICIOUS_PATTERNS
        ),

        "summary": {

            "files_checked": total_files,

            "target_columns_found": (
                total_target_columns
            ),

            "suspicious_columns_found": (
                total_suspicious_columns
            ),

            "duplicate_feature_rows": (
                total_duplicate_rows
            ),

            "status": overall_status
        },

        "files": results,

        "model_training_rule": (
            "Target/label columns must be "
            "excluded from model features."
        ),

        "duplicate_note": (
            "Duplicate feature rows are reported "
            "for investigation but are not "
            "automatically classified as leakage."
        )
    }

    # ========================================================
    # SAVE REPORT
    # ========================================================

    os.makedirs(
        os.path.dirname(
            OUTPUT_FILE
        ),
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
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL LEAKAGE AUDIT")
    print("=" * 70)

    print(
        f"Files checked              : "
        f"{total_files}"
    )

    print(
        f"Target columns found       : "
        f"{total_target_columns}"
    )

    print(
        f"Suspicious columns found   : "
        f"{total_suspicious_columns}"
    )

    print(
        f"Duplicate feature rows     : "
        f"{total_duplicate_rows:,}"
    )

    print(
        f"Overall status             : "
        f"{overall_status}"
    )

    print(
        f"Report saved to            : "
        f"{OUTPUT_FILE}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()