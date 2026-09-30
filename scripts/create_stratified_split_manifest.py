import os
import json
import hashlib
from collections import defaultdict

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = "data/processed"

OUTPUT_FILE = (
    "data/metadata/stratified_split_manifest.json"
)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42
CHUNK_SIZE = 100_000

# Target/label columns.
# Comparison is performed case-insensitively.
TARGET_COLUMNS = {
    "label",
    "attack_cat",
    "canonical_label",
    "binary_label",
    "original_label",
}

DATASETS = {
    "cicids2017": "data/processed/cicids2017",
    "cicids2018": "data/processed/cicids2018",
}


# ============================================================
# HELPERS
# ============================================================

def normalize_column_name(column):
    """
    Normalize a column name for case-insensitive comparison.
    """

    return str(column).strip().lower()


def normalize_value(value):
    """
    Convert values into deterministic strings for fingerprinting.
    """

    if pd.isna(value):
        return "<NA>"

    if isinstance(value, float):
        return f"{value:.12g}"

    return str(value).strip()


def make_fingerprint(row, feature_columns):
    """
    Create a deterministic SHA-256 fingerprint from feature values.

    Target columns are excluded before this function is called.

    Therefore, identical feature rows receive the same fingerprint
    regardless of their target/label values.
    """

    values = [
        normalize_value(row[col])
        for col in feature_columns
    ]

    raw = "\x1f".join(values)

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def get_csv_files(directory):
    """
    Return valid CSV files from a dataset directory.

    AppleDouble files such as ._filename.csv are ignored.
    """

    files = []

    for filename in sorted(os.listdir(directory)):

        if not filename.endswith(".csv"):
            continue

        if filename.startswith("._"):
            continue

        files.append(
            os.path.join(directory, filename)
        )

    return files


# ============================================================
# PASS 1
# DISCOVER SCHEMA
# ============================================================

def discover_columns(csv_files):
    """
    Discover all columns present across the dataset files.
    """

    all_columns = set()

    for file_path in csv_files:

        df = pd.read_csv(
            file_path,
            nrows=5
        )

        all_columns.update(df.columns)

    return sorted(all_columns)


# ============================================================
# FEATURE COLUMN SELECTION
# ============================================================

def get_feature_columns(all_columns):
    """
    Select feature columns while excluding all target columns.

    IMPORTANT:
    Comparison is case-insensitive.

    Therefore:
        Label
        label
        LABEL

    are all treated as the same target column.
    """

    feature_columns = []

    excluded_columns = []

    for column in all_columns:

        normalized = normalize_column_name(
            column
        )

        if normalized in TARGET_COLUMNS:

            excluded_columns.append(column)

        else:

            feature_columns.append(column)

    return feature_columns, excluded_columns


# ============================================================
# PASS 2
# CREATE DUPLICATE GROUPS
# ============================================================

def collect_duplicate_groups(
    csv_files,
    feature_columns
):
    """
    Scan all rows and create duplicate feature groups.

    Each fingerprint represents one unique feature vector.

    Target columns are NOT included in the fingerprint.
    """

    groups = {}

    label_counts = defaultdict(int)

    total_rows = 0

    for file_path in csv_files:

        print(
            f"Scanning: {file_path}"
        )

        for chunk in pd.read_csv(
            file_path,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ):

            # ------------------------------------------------
            # Validate canonical label
            # ------------------------------------------------

            if "canonical_label" not in chunk.columns:

                raise ValueError(
                    f"'canonical_label' missing in "
                    f"{file_path}"
                )

            # ------------------------------------------------
            # Validate feature columns
            # ------------------------------------------------

            missing_features = [
                col
                for col in feature_columns
                if col not in chunk.columns
            ]

            if missing_features:

                raise ValueError(
                    f"Missing feature columns in "
                    f"{file_path}: "
                    f"{missing_features[:10]}"
                )

            # ------------------------------------------------
            # Process rows
            # ------------------------------------------------

            for _, row in chunk.iterrows():

                label = str(
                    row["canonical_label"]
                ).strip()

                fingerprint = make_fingerprint(
                    row,
                    feature_columns
                )

                total_rows += 1

                label_counts[label] += 1

                # ------------------------------------------------
                # Create or update duplicate group
                # ------------------------------------------------

                if fingerprint not in groups:

                    groups[fingerprint] = {
                        "label": label,
                        "count": 0,
                    }

                groups[fingerprint]["count"] += 1

    return (
        groups,
        total_rows,
        label_counts
    )


# ============================================================
# DUPLICATE-AWARE STRATIFIED ASSIGNMENT
# ============================================================

def assign_groups(groups):
    """
    Assign every duplicate group completely to exactly
    one split.

    Splits:
        train       ≈ 70%
        validation  ≈ 15%
        test        ≈ 15%

    Groups belonging to the same feature fingerprint
    are NEVER divided between splits.

    Assignment is deterministic.
    """

    by_label = defaultdict(list)

    # --------------------------------------------------------
    # Group fingerprints by canonical label
    # --------------------------------------------------------

    for fingerprint, info in groups.items():

        by_label[info["label"]].append(
            (
                fingerprint,
                info["count"]
            )
        )

    assignments = {}

    # --------------------------------------------------------
    # Process each class separately
    # --------------------------------------------------------

    for label in sorted(by_label):

        label_groups = by_label[label]

        # ----------------------------------------------------
        # Deterministic ordering
        #
        # Larger duplicate groups are handled first.
        # Fingerprint is used as deterministic tie-breaker.
        # ----------------------------------------------------

        label_groups.sort(
            key=lambda x: (
                -x[1],
                x[0]
            )
        )

        total = sum(
            count
            for _, count in label_groups
        )

        # ----------------------------------------------------
        # Target row counts
        # ----------------------------------------------------

        target_train = (
            total * TRAIN_RATIO
        )

        target_val = (
            total * VAL_RATIO
        )

        target_test = (
            total * TEST_RATIO
        )

        current = {
            "train": 0,
            "validation": 0,
            "test": 0,
        }

        targets = {
            "train": target_train,
            "validation": target_val,
            "test": target_test,
        }

        # ----------------------------------------------------
        # Assign each entire duplicate group
        # ----------------------------------------------------

        for fingerprint, count in label_groups:

            # Select split with the largest remaining need.
            split = max(
                targets,
                key=lambda s:
                    targets[s] - current[s]
            )

            assignments[fingerprint] = split

            current[split] += count

    return assignments


# ============================================================
# BUILD MANIFEST
# ============================================================

def build_dataset_manifest(
    dataset_name,
    directory
):
    """
    Build the stratified split manifest for one dataset.
    """

    csv_files = get_csv_files(
        directory
    )

    if not csv_files:

        raise ValueError(
            f"No CSV files found in {directory}"
        )

    print()
    print("=" * 70)
    print(
        f"DATASET: {dataset_name}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Discover schema
    # --------------------------------------------------------

    all_columns = discover_columns(
        csv_files
    )

    # --------------------------------------------------------
    # Select feature columns
    # --------------------------------------------------------

    feature_columns, excluded_columns = (
        get_feature_columns(
            all_columns
        )
    )

    print(
        f"Files: {len(csv_files)}"
    )

    print(
        f"All columns discovered: "
        f"{len(all_columns)}"
    )

    print(
        f"Feature columns: "
        f"{len(feature_columns)}"
    )

    print(
        f"Excluded target columns: "
        f"{len(excluded_columns)}"
    )

    if excluded_columns:

        print(
            "Excluded:"
        )

        for column in excluded_columns:

            print(
                f"  - {column}"
            )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    remaining_target_columns = [
        col
        for col in feature_columns
        if normalize_column_name(col)
        in TARGET_COLUMNS
    ]

    if remaining_target_columns:

        raise RuntimeError(
            "Target-column leakage detected. "
            "These target columns are still present "
            f"in feature_columns: "
            f"{remaining_target_columns}"
        )

    # --------------------------------------------------------
    # Create duplicate groups
    # --------------------------------------------------------

    (
        groups,
        total_rows,
        label_counts
    ) = collect_duplicate_groups(
        csv_files,
        feature_columns
    )

    print(
        f"Rows: {total_rows:,}"
    )

    print(
        f"Unique feature groups: "
        f"{len(groups):,}"
    )

    duplicate_rows = (
        total_rows
        - len(groups)
    )

    print(
        f"Duplicate rows: "
        f"{duplicate_rows:,}"
    )

    # --------------------------------------------------------
    # Assign duplicate groups
    # --------------------------------------------------------

    assignments = assign_groups(
        groups
    )

    # --------------------------------------------------------
    # Calculate split counts
    # --------------------------------------------------------

    split_counts = {
        "train": 0,
        "validation": 0,
        "test": 0,
    }

    for fingerprint, split in assignments.items():

        split_counts[split] += (
            groups[fingerprint]["count"]
        )

    # --------------------------------------------------------
    # Calculate percentages
    # --------------------------------------------------------

    split_percentages = {}

    for split, count in split_counts.items():

        split_percentages[split] = (
            count / total_rows * 100
            if total_rows > 0
            else 0
        )

    # --------------------------------------------------------
    # Count duplicate groups per split
    # --------------------------------------------------------

    split_group_counts = {
        "train": 0,
        "validation": 0,
        "test": 0,
    }

    for fingerprint, split in assignments.items():

        split_group_counts[split] += 1

    # --------------------------------------------------------
    # Build file manifest
    # --------------------------------------------------------

    manifest_files = []

    for file_path in csv_files:

        manifest_files.append(
            {
                "file": file_path,
                "dataset": dataset_name,
            }
        )

    # --------------------------------------------------------
    # Build result
    # --------------------------------------------------------

    return {
        "dataset": dataset_name,

        "files": manifest_files,

        "total_rows": total_rows,

        "unique_feature_groups": len(
            groups
        ),

        "duplicate_rows": duplicate_rows,

        "feature_columns": feature_columns,

        "excluded_target_columns": (
            excluded_columns
        ),

        "split_counts": split_counts,

        "split_percentages": (
            split_percentages
        ),

        "split_group_counts": (
            split_group_counts
        ),

        "label_counts": dict(
            sorted(
                label_counts.items()
            )
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Create metadata directory
    # --------------------------------------------------------

    os.makedirs(
        "data/metadata",
        exist_ok=True
    )

    # --------------------------------------------------------
    # Validate ratios
    # --------------------------------------------------------

    ratio_sum = (
        TRAIN_RATIO
        + VAL_RATIO
        + TEST_RATIO
    )

    if abs(ratio_sum - 1.0) > 1e-9:

        raise ValueError(
            "Train/validation/test ratios must "
            f"sum to 1.0. Current sum: "
            f"{ratio_sum}"
        )

    # --------------------------------------------------------
    # Result structure
    # --------------------------------------------------------

    result = {

        "status": "PASS",

        "method": {

            "type":
                "duplicate_aware_stratified",

            "train_ratio":
                TRAIN_RATIO,

            "validation_ratio":
                VAL_RATIO,

            "test_ratio":
                TEST_RATIO,

            "random_seed":
                RANDOM_SEED,

            "chunk_size":
                CHUNK_SIZE,

            "target_columns_case_sensitive":
                False,

            "duplicate_policy":
                "keep_duplicate_feature_groups_together",
        },

        "datasets": {},
    }

    # --------------------------------------------------------
    # Process datasets
    # --------------------------------------------------------

    for dataset_name, directory in DATASETS.items():

        result["datasets"][dataset_name] = (
            build_dataset_manifest(
                dataset_name,
                directory
            )
        )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "SPLIT MANIFEST CREATED"
    )
    print("=" * 70)

    print(
        f"Output: {OUTPUT_FILE}"
    )

    for dataset_name, info in (
        result["datasets"].items()
    ):

        print()
        print(
            dataset_name
        )

        print(
            f"Total rows: "
            f"{info['total_rows']:,}"
        )

        print(
            f"Unique feature groups: "
            f"{info['unique_feature_groups']:,}"
        )

        print(
            f"Duplicate rows: "
            f"{info['duplicate_rows']:,}"
        )

        print(
            f"Feature columns: "
            f"{len(info['feature_columns'])}"
        )

        print(
            f"Excluded targets: "
            f"{info['excluded_target_columns']}"
        )

        print()

        print(
            f"Train:      "
            f"{info['split_counts']['train']:,}"
            f" "
            f"({info['split_percentages']['train']:.4f}%)"
        )

        print(
            f"Validation: "
            f"{info['split_counts']['validation']:,}"
            f" "
            f"({info['split_percentages']['validation']:.4f}%)"
        )

        print(
            f"Test:       "
            f"{info['split_counts']['test']:,}"
            f" "
            f"({info['split_percentages']['test']:.4f}%)"
        )

        print()

        print(
            f"Train groups: "
            f"{info['split_group_counts']['train']:,}"
        )

        print(
            f"Validation groups: "
            f"{info['split_group_counts']['validation']:,}"
        )

        print(
            f"Test groups: "
            f"{info['split_group_counts']['test']:,}"
        )

    print()
    print("=" * 70)
    print(
        "STATUS: PASS"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()