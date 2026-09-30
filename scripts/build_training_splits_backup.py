#!/usr/bin/env python3

"""
CyberSecAI
Build Actual Train / Validation / Test Split Files

IMPORTANT:
This script intentionally reproduces the SAME fingerprinting and
duplicate-group assignment logic used by:

    scripts/create_stratified_split_manifest.py

It therefore uses:

    normalize_value():
        float -> "{value:.12g}"

    make_fingerprint():
        SHA-256 over feature values joined by \\x1f

    feature columns:
        union of all columns across the dataset

    conflicting feature-label groups:
        first-seen canonical label, exactly like the original
        split-manifest generator

The goal is to create physical split CSV files without changing
the validated Step 9 methodology.
"""

import os
import json
import hashlib
from collections import defaultdict

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = "data/processed"

OUTPUT_ROOT = "data/splits"

METADATA_ROOT = "data/metadata"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

CHUNK_SIZE = 100_000

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
# EXACT HELPERS FROM VALIDATED SPLIT GENERATOR
# ============================================================

def normalize_column_name(column):
    """
    Normalize a column name for case-insensitive comparison.
    """

    return str(column).strip().lower()


def normalize_value(value):
    """
    EXACTLY matches create_stratified_split_manifest.py.

    Do NOT change .12g.
    """

    if pd.isna(value):
        return "<NA>"

    if isinstance(value, float):
        return f"{value:.12g}"

    return str(value).strip()


def make_fingerprint(row, feature_columns):
    """
    EXACTLY matches create_stratified_split_manifest.py.
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
    Return valid CSV files.

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
# DISCOVER DATASET SCHEMA
# ============================================================

def discover_columns(csv_files):
    """
    Discover the UNION of all columns across the dataset.

    This is exactly how the validated manifest generator
    determines feature columns.
    """

    all_columns = set()

    for file_path in csv_files:

        df = pd.read_csv(
            file_path,
            nrows=5
        )

        all_columns.update(
            df.columns
        )

    return sorted(all_columns)


def get_feature_columns(all_columns):
    """
    Select feature columns while excluding target columns.
    """

    feature_columns = []

    excluded_columns = []

    for column in all_columns:

        normalized = normalize_column_name(
            column
        )

        if normalized in TARGET_COLUMNS:

            excluded_columns.append(
                column
            )

        else:

            feature_columns.append(
                column
            )

    return (
        feature_columns,
        excluded_columns
    )


# ============================================================
# PASS 1
# CREATE EXACT SAME DUPLICATE GROUPS
# ============================================================

def collect_duplicate_groups(
    csv_files,
    feature_columns
):
    """
    EXACTLY reproduce the grouping logic of
    create_stratified_split_manifest.py.
    """

    groups = {}

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
            # canonical_label must exist
            # ------------------------------------------------

            if "canonical_label" not in chunk.columns:

                raise ValueError(
                    f"'canonical_label' missing in "
                    f"{file_path}"
                )

            # ------------------------------------------------
            # Validate every union feature exists
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
            # IMPORTANT:
            #
            # Use iterrows exactly like original generator.
            # This preserves the original fingerprint behavior.
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

                if fingerprint not in groups:

                    # IMPORTANT:
                    # First-seen label is preserved.
                    groups[fingerprint] = {
                        "label": label,
                        "count": 0,
                    }

                groups[fingerprint]["count"] += 1

    return (
        groups,
        total_rows
    )


# ============================================================
# EXACT SAME GROUP ASSIGNMENT
# ============================================================

def assign_groups(groups):
    """
    EXACTLY reproduce assign_groups() from the validated
    split-manifest generator.
    """

    by_label = defaultdict(list)

    # --------------------------------------------------------
    # Group fingerprints by FIRST-SEEN canonical label
    # --------------------------------------------------------

    for fingerprint, info in groups.items():

        by_label[
            info["label"]
        ].append(
            (
                fingerprint,
                info["count"]
            )
        )

    assignments = {}

    # --------------------------------------------------------
    # Process labels in sorted order
    # --------------------------------------------------------

    for label in sorted(by_label):

        label_groups = by_label[label]

        # ----------------------------------------------------
        # EXACT deterministic ordering
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
        # EXACT same largest-remaining-need assignment
        # ----------------------------------------------------

        for fingerprint, count in label_groups:

            split = max(
                targets,
                key=lambda s:
                    targets[s] - current[s]
            )

            assignments[
                fingerprint
            ] = split

            current[split] += count

    return assignments


# ============================================================
# REBUILD THE EXACT VALIDATED ASSIGNMENT
# ============================================================

def build_assignment(
    dataset_name,
    dataset_directory
):
    """
    Reconstruct the exact feature-group assignment used by
    create_stratified_split_manifest.py.
    """

    csv_files = get_csv_files(
        dataset_directory
    )

    if not csv_files:

        raise ValueError(
            f"No CSV files found in "
            f"{dataset_directory}"
        )

    print()
    print("=" * 70)
    print(
        f"BUILDING ASSIGNMENT: {dataset_name}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Discover union schema
    # --------------------------------------------------------

    all_columns = discover_columns(
        csv_files
    )

    feature_columns, excluded_columns = (
        get_feature_columns(
            all_columns
        )
    )

    print(
        f"CSV files found: {len(csv_files)}"
    )

    print(
        f"All columns: {len(all_columns)}"
    )

    print(
        f"Feature columns: "
        f"{len(feature_columns)}"
    )

    print(
        "Excluded target columns:",
        excluded_columns
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    leaked_targets = [
        col
        for col in feature_columns
        if normalize_column_name(col)
        in TARGET_COLUMNS
    ]

    if leaked_targets:

        raise RuntimeError(
            "Target columns remain in feature set: "
            f"{leaked_targets}"
        )

    # --------------------------------------------------------
    # Discover groups
    # --------------------------------------------------------

    groups, total_rows = (
        collect_duplicate_groups(
            csv_files,
            feature_columns
        )
    )

    unique_groups = len(groups)

    duplicate_rows = (
        total_rows - unique_groups
    )

    print()
    print(
        f"Total rows: {total_rows:,}"
    )

    print(
        f"Unique feature groups: "
        f"{unique_groups:,}"
    )

    print(
        f"Duplicate rows: "
        f"{duplicate_rows:,}"
    )

    # --------------------------------------------------------
    # ASSIGN
    # --------------------------------------------------------

    assignments = assign_groups(
        groups
    )

    # --------------------------------------------------------
    # Split group counts
    # --------------------------------------------------------

    split_group_counts = {
        "train": 0,
        "validation": 0,
        "test": 0,
    }

    split_row_counts = {
        "train": 0,
        "validation": 0,
        "test": 0,
    }

    for fingerprint, split in assignments.items():

        split_group_counts[split] += 1

        split_row_counts[split] += (
            groups[fingerprint]["count"]
        )

    print()
    print(
        "Feature groups by split:"
    )

    for split in (
        "train",
        "validation",
        "test"
    ):

        print(
            f"  {split:12s}: "
            f"{split_group_counts[split]:,}"
        )

    print()
    print(
        "Rows by split:"
    )

    for split in (
        "train",
        "validation",
        "test"
    ):

        percentage = (
            split_row_counts[split]
            / total_rows
            * 100
        )

        print(
            f"  {split:12s}: "
            f"{split_row_counts[split]:,} "
            f"({percentage:.6f}%)"
        )

    # --------------------------------------------------------
    # Conservation check
    # --------------------------------------------------------

    assigned_rows = sum(
        split_row_counts.values()
    )

    if assigned_rows != total_rows:

        raise RuntimeError(
            "Split row conservation failed.\n"
            f"Original: {total_rows:,}\n"
            f"Assigned: {assigned_rows:,}"
        )

    print()
    print(
        "Row conservation: PASS"
    )

    return {
        "csv_files": csv_files,
        "feature_columns": feature_columns,
        "excluded_target_columns": (
            excluded_columns
        ),
        "groups": groups,
        "assignments": assignments,
        "total_rows": total_rows,
        "unique_groups": unique_groups,
        "duplicate_rows": duplicate_rows,
        "split_group_counts": (
            split_group_counts
        ),
        "split_row_counts": (
            split_row_counts
        ),
    }


# ============================================================
# WRITE ACTUAL RAW SPLIT FILES
# ============================================================

def write_split_files(
    dataset_name,
    info
):
    """
    Write actual split CSV files.

    IMPORTANT:
    Raw feature columns are retained here.

    Labels are retained because later preprocessing/model
    stages need them.

    No rows are deleted.
    No labels are changed.
    """

    dataset_output = os.path.join(
        OUTPUT_ROOT,
        dataset_name
    )

    split_directories = {}

    for split in (
        "train",
        "validation",
        "test"
    ):

        split_dir = os.path.join(
            dataset_output,
            split
        )

        os.makedirs(
            split_dir,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Remove only previously generated part files.
        # ----------------------------------------------------

        for filename in os.listdir(
            split_dir
        ):

            if filename.startswith(
                "part_"
            ) and filename.endswith(
                ".csv"
            ):

                os.remove(
                    os.path.join(
                        split_dir,
                        filename
                    )
                )

        split_directories[
            split
        ] = split_dir

    counters = {
        "train": 0,
        "validation": 0,
        "test": 0,
    }

    part_numbers = {
        "train": 0,
        "validation": 0,
        "test": 0,
    }

    # --------------------------------------------------------
    # Process original files
    # --------------------------------------------------------

    for file_path in info["csv_files"]:

        print()
        print(
            f"Writing from: "
            f"{os.path.basename(file_path)}"
        )

        for chunk in pd.read_csv(
            file_path,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ):

            fingerprints = []

            for _, row in chunk.iterrows():

                fingerprints.append(
                    make_fingerprint(
                        row,
                        info["feature_columns"]
                    )
                )

            indices = {
                "train": [],
                "validation": [],
                "test": [],
            }

            for idx, fingerprint in enumerate(
                fingerprints
            ):

                if fingerprint not in info[
                    "assignments"
                ]:

                    raise RuntimeError(
                        "Fingerprint missing from "
                        "assignment."
                    )

                split = info[
                    "assignments"
                ][fingerprint]

                indices[
                    split
                ].append(idx)

            # ------------------------------------------------
            # Write each split
            # ------------------------------------------------

            for split in (
                "train",
                "validation",
                "test"
            ):

                rows = indices[split]

                if not rows:
                    continue

                output_df = chunk.iloc[
                    rows
                ].copy()

                part_number = (
                    part_numbers[split]
                )

                output_file = os.path.join(
                    split_directories[split],
                    f"part_{part_number:04d}.csv"
                )

                output_df.to_csv(
                    output_file,
                    index=False
                )

                counters[split] += (
                    len(output_df)
                )

                part_numbers[split] += 1

    print()
    print(
        "Actual rows written:"
    )

    for split in (
        "train",
        "validation",
        "test"
    ):

        percentage = (
            counters[split]
            / info["total_rows"]
            * 100
        )

        print(
            f"  {split:12s}: "
            f"{counters[split]:,} "
            f"({percentage:.6f}%)"
        )

    # --------------------------------------------------------
    # Final conservation check
    # --------------------------------------------------------

    written_total = sum(
        counters.values()
    )

    if written_total != info[
        "total_rows"
    ]:

        raise RuntimeError(
            "Written row count does not match "
            "original row count.\n"
            f"Original: {info['total_rows']:,}\n"
            f"Written:  {written_total:,}"
        )

    print()
    print(
        "Written-row conservation: PASS"
    )

    return counters


# ============================================================
# MANIFEST
# ============================================================

def save_output_manifest(
    dataset_name,
    info,
    written_counts
):
    """
    Save metadata describing the actual split files.
    """

    dataset_output = os.path.join(
        OUTPUT_ROOT,
        dataset_name
    )

    split_files = {}

    for split in (
        "train",
        "validation",
        "test"
    ):

        split_dir = os.path.join(
            dataset_output,
            split
        )

        split_files[split] = [
            os.path.relpath(
                os.path.join(
                    split_dir,
                    filename
                )
            )
            for filename in sorted(
                os.listdir(split_dir)
            )
            if filename.startswith(
                "part_"
            )
            and filename.endswith(
                ".csv"
            )
        ]

    manifest = {
        "dataset": dataset_name,

        "method": {
            "type":
                "duplicate_aware_stratified",

            "fingerprint":
                "SHA-256",

            "normalization":
                "exactly_matches_create_stratified_split_manifest",

            "train_ratio":
                TRAIN_RATIO,

            "validation_ratio":
                VAL_RATIO,

            "test_ratio":
                TEST_RATIO,

            "duplicate_policy":
                "keep_duplicate_feature_groups_together",

            "conflicting_label_policy":
                "first_seen_label_for_stratification_only",
        },

        "total_rows":
            info["total_rows"],

        "unique_feature_groups":
            info["unique_groups"],

        "duplicate_rows":
            info["duplicate_rows"],

        "feature_column_count":
            len(info["feature_columns"]),

        "excluded_target_columns":
            info["excluded_target_columns"],

        "split_group_counts":
            info["split_group_counts"],

        "split_row_counts":
            info["split_row_counts"],

        "actual_written_counts":
            written_counts,

        "split_files":
            split_files,
    }

    manifest_file = os.path.join(
        METADATA_ROOT,
        f"{dataset_name}_training_split_manifest.json"
    )

    os.makedirs(
        METADATA_ROOT,
        exist_ok=True
    )

    with open(
        manifest_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2
        )

    print()
    print(
        "Manifest saved:"
    )
    print(
        manifest_file
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "CYBERSECAI TRAINING SPLIT BUILDER"
    )
    print("=" * 70)

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
            "Split ratios must sum to 1.0."
        )

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_ROOT,
        exist_ok=True
    )

    os.makedirs(
        METADATA_ROOT,
        exist_ok=True
    )

    all_results = {}

    # --------------------------------------------------------
    # Process datasets
    # --------------------------------------------------------

    for dataset_name, directory in DATASETS.items():

        info = build_assignment(
            dataset_name,
            directory
        )

        written_counts = (
            write_split_files(
                dataset_name,
                info
            )
        )

        save_output_manifest(
            dataset_name,
            info,
            written_counts
        )

        all_results[
            dataset_name
        ] = {
            "total_rows":
                info["total_rows"],

            "unique_groups":
                info["unique_groups"],

            "duplicate_rows":
                info["duplicate_rows"],

            "split_rows":
                written_counts,
        }

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "FINAL SUMMARY"
    )
    print("=" * 70)

    for dataset_name, result in (
        all_results.items()
    ):

        print()
        print(
            dataset_name
        )

        print(
            "Rows:",
            f"{result['total_rows']:,}"
        )

        print(
            "Unique groups:",
            f"{result['unique_groups']:,}"
        )

        print(
            "Duplicate rows:",
            f"{result['duplicate_rows']:,}"
        )

        for split, count in (
            result["split_rows"].items()
        ):

            percentage = (
                count
                / result["total_rows"]
                * 100
            )

            print(
                f"{split:12s}: "
                f"{count:,} "
                f"({percentage:.6f}%)"
            )

    print()
    print("=" * 70)
    print(
        "TRAINING SPLITS CREATED SUCCESSFULLY"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()