#!/usr/bin/env python3

"""
CyberSecAI - Training / Validation / Test Split Builder

Dataset policy
--------------

CIC-IDS2017
    Feature-engineered data
    70% train
    15% validation
    15% test
    Duplicate feature groups remain in the same split.

CIC-IDS2018
    Feature-engineered data
    70% train
    15% validation
    15% test
    Duplicate feature groups remain in the same split.

UNSW-NB15
    Official training file:
        85% train
        15% validation

    Official testing file:
        100% test

    The official UNSW train/test boundary is preserved.

Important
---------
Splitting is performed using:

    data/features/

NOT:

    data/processed/

Labels/target columns are excluded from feature fingerprints.
Original rows and columns are preserved in the generated split files.
"""

from pathlib import Path
from collections import defaultdict
import hashlib
import json
import math
import shutil

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_ROOT = ROOT / "data" / "features"
OUTPUT_ROOT = ROOT / "data" / "splits"
METADATA_ROOT = ROOT / "data" / "metadata"

CHUNK_SIZE = 100_000


# ------------------------------------------------------------
# CIC split ratios
# ------------------------------------------------------------

CIC_TRAIN_RATIO = 0.70
CIC_VAL_RATIO = 0.15
CIC_TEST_RATIO = 0.15


# ------------------------------------------------------------
# UNSW official training split
# ------------------------------------------------------------

UNSW_TRAIN_RATIO = 0.85
UNSW_VAL_RATIO = 0.15


# ------------------------------------------------------------
# Columns that must NOT participate in fingerprints
# ------------------------------------------------------------

TARGET_COLUMNS = {
    "label",
    "attack_cat",
    "canonical_label",
    "binary_label",
    "original_label",
}


# ------------------------------------------------------------
# Dataset locations
# ------------------------------------------------------------

DATASETS = {
    "cicids2017": FEATURE_ROOT / "cicids2017",
    "cicids2018": FEATURE_ROOT / "cicids2018",
    "unswnb15": FEATURE_ROOT / "unswnb15",
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_column_name(name):
    """
    Normalize column names for target-column comparison.
    """

    return str(name).strip().lower()


def normalize_value(value):
    """
    Normalize values before fingerprint generation.

    Numeric values use 12 significant digits.
    Missing values use <NA>.
    Positive/negative infinity get fixed tokens.
    """

    if pd.isna(value):
        return "<NA>"

    if isinstance(value, bool):
        return "1" if value else "0"

    if isinstance(value, (int, float)):

        if isinstance(value, float):

            if math.isnan(value):
                return "<NA>"

            if math.isinf(value):

                if value > 0:
                    return "<INF>"

                return "<-INF>"

        return format(value, ".12g")

    return str(value).strip()


def make_fingerprint(row, feature_columns):
    """
    Generate a SHA-256 fingerprint from feature values.

    Target/label columns are intentionally excluded.
    """

    values = []

    for column in feature_columns:

        values.append(
            normalize_value(row[column])
        )

    payload = "\x1f".join(values)

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def get_csv_files(dataset_path):
    """
    Return CSV files from a dataset directory.

    macOS ._* metadata files are ignored.
    """

    if not dataset_path.exists():

        raise FileNotFoundError(
            f"Dataset directory does not exist: "
            f"{dataset_path}"
        )

    files = sorted(
        path
        for path in dataset_path.glob("*.csv")
        if not path.name.startswith("._")
    )

    if not files:

        raise FileNotFoundError(
            f"No CSV files found in: "
            f"{dataset_path}"
        )

    return files


# ============================================================
# SCHEMA DISCOVERY
# ============================================================

def discover_union_columns(csv_files):
    """
    Discover the union of columns across CSV files.

    The order follows first appearance.
    """

    columns = []
    seen = set()

    for csv_file in csv_files:

        header = pd.read_csv(
            csv_file,
            nrows=0
        )

        for column in header.columns:

            if column not in seen:

                seen.add(column)
                columns.append(column)

    return columns


def get_target_columns(columns):
    """
    Identify target/label columns actually present.
    """

    targets = set()

    for column in columns:

        normalized = normalize_column_name(
            column
        )

        if normalized in TARGET_COLUMNS:

            targets.add(column)

    return targets


def get_feature_columns(columns):
    """
    Return columns used for duplicate fingerprints.

    All label/target columns are excluded.
    """

    target_columns = get_target_columns(
        columns
    )

    return [
        column
        for column in columns
        if column not in target_columns
    ]


# ============================================================
# DUPLICATE GROUP COLLECTION
# ============================================================

def collect_duplicate_groups(
    csv_files,
    feature_columns
):
    """
    Collect rows into feature-identical groups.

    Each unique feature fingerprint represents one group.
    """

    groups = {}

    total_rows = 0

    for csv_file in csv_files:

        print(
            f"  Reading: {csv_file.name}"
        )

        for chunk in pd.read_csv(
            csv_file,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ):

            # Validate feature columns.
            missing = [
                column
                for column in feature_columns
                if column not in chunk.columns
            ]

            if missing:

                raise ValueError(
                    f"{csv_file.name} is missing "
                    f"feature columns: "
                    f"{missing[:10]}"
                )

            for _, row in chunk.iterrows():

                total_rows += 1

                fingerprint = make_fingerprint(
                    row,
                    feature_columns
                )

                # ------------------------------------------------
                # Canonical label
                # ------------------------------------------------

                canonical_label = str(
                    row.get(
                        "canonical_label",
                        ""
                    )
                ).strip()

                if not canonical_label:

                    canonical_label = str(
                        row.get(
                            "original_label",
                            ""
                        )
                    ).strip()

                if not canonical_label:

                    canonical_label = "UNKNOWN"

                # ------------------------------------------------
                # First occurrence creates group.
                # ------------------------------------------------

                if fingerprint not in groups:

                    groups[fingerprint] = {
                        "fingerprint": fingerprint,
                        "count": 0,
                        "canonical_label":
                            canonical_label,
                    }

                groups[fingerprint]["count"] += 1

    duplicate_rows = sum(
        group["count"] - 1
        for group in groups.values()
        if group["count"] > 1
    )

    return (
        groups,
        total_rows,
        duplicate_rows
    )


# ============================================================
# GROUP ASSIGNMENT
# ============================================================

def assign_groups(
    groups,
    split_ratios
):
    """
    Assign complete duplicate groups to splits.

    Groups are stratified by canonical label.

    Within each label:

        1. Larger groups are processed first.
        2. The split with the largest remaining target
           receives the group.

    This prevents identical feature groups from being
    distributed across train/validation/test.
    """

    split_names = list(
        split_ratios.keys()
    )

    total_rows = sum(
        group["count"]
        for group in groups.values()
    )

    target_rows = {
        split:
            total_rows * ratio
        for split, ratio in split_ratios.items()
    }

    current_rows = {
        split: 0
        for split in split_names
    }

    assignment = {}

    label_groups = defaultdict(list)

    # --------------------------------------------------------
    # Group by canonical label.
    # --------------------------------------------------------

    for fingerprint, group in groups.items():

        label_groups[
            group["canonical_label"]
        ].append(group)

    # --------------------------------------------------------
    # Process each label separately.
    # --------------------------------------------------------

    for label in sorted(
        label_groups.keys()
    ):

        current_label_groups = sorted(
            label_groups[label],
            key=lambda item: (
                -item["count"],
                item["fingerprint"],
            )
        )

        for group in current_label_groups:

            # Remaining target for each split.
            remaining = {
                split:
                    target_rows[split]
                    - current_rows[split]
                for split in split_names
            }

            # Select the split with largest
            # remaining requirement.
            selected_split = max(
                split_names,
                key=lambda split: (
                    remaining[split],
                    -current_rows[split]
                )
            )

            assignment[
                group["fingerprint"]
            ] = selected_split

            current_rows[
                selected_split
            ] += group["count"]

    return (
        assignment,
        current_rows,
        target_rows
    )


# ============================================================
# BUILD GROUP ASSIGNMENT
# ============================================================

def build_group_assignment(
    csv_files,
    split_ratios,
    dataset_name
):
    """
    Build duplicate-aware assignment.
    """

    print()
    print("=" * 70)
    print(
        f"BUILDING GROUP ASSIGNMENT: "
        f"{dataset_name}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Discover schema
    # --------------------------------------------------------

    all_columns = discover_union_columns(
        csv_files
    )

    feature_columns = get_feature_columns(
        all_columns
    )

    print(
        f"  Total schema columns: "
        f"{len(all_columns)}"
    )

    print(
        f"  Feature columns:      "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Collect groups
    # --------------------------------------------------------

    (
        groups,
        total_rows,
        duplicate_rows
    ) = collect_duplicate_groups(
        csv_files,
        feature_columns
    )

    print(
        f"  Total rows:           "
        f"{total_rows:,}"
    )

    print(
        f"  Unique groups:        "
        f"{len(groups):,}"
    )

    print(
        f"  Duplicate rows:       "
        f"{duplicate_rows:,}"
    )

    # --------------------------------------------------------
    # Assign groups
    # --------------------------------------------------------

    (
        assignment,
        actual_counts,
        target_counts
    ) = assign_groups(
        groups,
        split_ratios
    )

    print()
    print(
        "  Split targets / actual:"
    )

    for split in split_ratios:

        target = target_counts[
            split
        ]

        actual = actual_counts[
            split
        ]

        percentage = (
            actual / total_rows * 100
            if total_rows
            else 0
        )

        print(
            f"    {split:12s}: "
            f"{actual:,} rows "
            f"({percentage:.6f}%) "
            f"target={target:.2f}"
        )

    # --------------------------------------------------------
    # Assignment conservation
    # --------------------------------------------------------

    assigned_total = sum(
        actual_counts.values()
    )

    if assigned_total != total_rows:

        raise RuntimeError(
            f"Row conservation failed for "
            f"{dataset_name}: "
            f"{assigned_total} != "
            f"{total_rows}"
        )

    return {
        "assignment": assignment,
        "feature_columns": feature_columns,
        "all_columns": all_columns,
        "total_rows": total_rows,
        "unique_groups": len(groups),
        "duplicate_rows": duplicate_rows,
        "split_counts": actual_counts,
        "target_counts": target_counts,
    }


# ============================================================
# SAFE DIRECTORY CLEANUP
# ============================================================

def safe_remove_directory(path):
    """
    Safely remove a directory.

    ignore_errors=True is intentional here because the
    project is stored on an external SD card and macOS
    filesystem operations can occasionally report that
    an entry disappeared during cleanup.
    """

    if path.exists():

        shutil.rmtree(
            path,
            ignore_errors=True
        )


def prepare_split_directories(
    output_dir,
    splits
):
    """
    Clean and recreate split directories.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for split in splits:

        split_dir = (
            output_dir / split
        )

        safe_remove_directory(
            split_dir
        )

        split_dir.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# WRITE GROUP-ASSIGNED SPLITS
# ============================================================

def write_group_assigned_splits(
    csv_files,
    assignment,
    feature_columns,
    output_dir,
    splits=None
):
    """
    Write rows according to fingerprint assignment.

    Original columns are preserved.

    Each chunk is written as a separate part file.
    """

    if splits is None:

        splits = [
            "train",
            "validation",
            "test"
        ]

    # --------------------------------------------------------
    # Prepare output directories.
    # --------------------------------------------------------

    prepare_split_directories(
        output_dir,
        splits
    )

    part_numbers = {
        split: 0
        for split in splits
    }

    written_counts = {
        split: 0
        for split in splits
    }

    print()
    print(
        "Writing physical split files..."
    )

    # --------------------------------------------------------
    # Process source files.
    # --------------------------------------------------------

    for csv_file in csv_files:

        print(
            f"  Processing: "
            f"{csv_file.name}"
        )

        for chunk in pd.read_csv(
            csv_file,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ):

            split_rows = {
                split: []
                for split in splits
            }

            # ------------------------------------------------
            # Assign every row using its fingerprint.
            # ------------------------------------------------

            for index, row in chunk.iterrows():

                fingerprint = make_fingerprint(
                    row,
                    feature_columns
                )

                if fingerprint not in assignment:

                    raise RuntimeError(
                        "Fingerprint missing from "
                        f"assignment: {fingerprint}"
                    )

                split = assignment[
                    fingerprint
                ]

                if split not in split_rows:

                    raise RuntimeError(
                        f"Invalid split assignment: "
                        f"{split}"
                    )

                split_rows[
                    split
                ].append(index)

            # ------------------------------------------------
            # Write rows.
            # ------------------------------------------------

            for split in splits:

                indices = split_rows[
                    split
                ]

                if not indices:
                    continue

                selected = chunk.loc[
                    indices
                ]

                part_numbers[
                    split
                ] += 1

                output_file = (
                    output_dir
                    / split
                    / (
                        f"part_"
                        f"{part_numbers[split]:04d}"
                        f".csv"
                    )
                )

                selected.to_csv(
                    output_file,
                    index=False
                )

                written_counts[
                    split
                ] += len(selected)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total_written = sum(
        written_counts.values()
    )

    print()
    print("Written rows:")

    for split in splits:

        print(
            f"  {split:12s}: "
            f"{written_counts[split]:,}"
        )

    print(
        f"  {'TOTAL':12s}: "
        f"{total_written:,}"
    )

    return written_counts


# ============================================================
# WRITE OFFICIAL UNSW TEST SET
# ============================================================

def write_official_test_file(
    testing_file,
    output_dir
):
    """
    Copy the official UNSW-NB15 testing file entirely
    into the test split.

    No random splitting is performed.
    """

    test_dir = (
        output_dir / "test"
    )

    safe_remove_directory(
        test_dir
    )

    test_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    written_rows = 0
    part_number = 0

    print()
    print(
        "Preserving official UNSW "
        "test file:"
    )

    print(
        f"  {testing_file.name}"
    )

    for chunk in pd.read_csv(
        testing_file,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        part_number += 1

        output_file = (
            test_dir
            / (
                f"part_"
                f"{part_number:04d}.csv"
            )
        )

        chunk.to_csv(
            output_file,
            index=False
        )

        written_rows += len(chunk)

    print(
        f"  Official test rows: "
        f"{written_rows:,}"
    )

    return written_rows


# ============================================================
# SAVE MANIFEST
# ============================================================

def save_manifest(
    dataset_name,
    manifest
):
    """
    Save split metadata manifest.
    """

    METADATA_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        METADATA_ROOT
        / (
            f"{dataset_name}"
            f"_training_split_manifest.json"
        )
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2
        )

    print()
    print(
        "Manifest saved:"
    )

    print(
        f"  {output_file}"
    )

    return output_file


# ============================================================
# PROCESS CIC DATASETS
# ============================================================

def process_cic_dataset(
    dataset_name,
    dataset_path
):
    """
    Process CIC-IDS2017 or CIC-IDS2018.
    """

    csv_files = get_csv_files(
        dataset_path
    )

    print()
    print(
        f"Dataset: {dataset_name}"
    )

    print(
        f"CSV files: {len(csv_files)}"
    )

    # --------------------------------------------------------
    # CIC 70 / 15 / 15
    # --------------------------------------------------------

    split_ratios = {
        "train": CIC_TRAIN_RATIO,
        "validation": CIC_VAL_RATIO,
        "test": CIC_TEST_RATIO,
    }

    result = build_group_assignment(
        csv_files,
        split_ratios,
        dataset_name
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_dir = (
        OUTPUT_ROOT
        / dataset_name
    )

    written_counts = (
        write_group_assigned_splits(
            csv_files,
            result["assignment"],
            result["feature_columns"],
            output_dir
        )
    )

    # --------------------------------------------------------
    # Final conservation check
    # --------------------------------------------------------

    expected_total = (
        result["total_rows"]
    )

    actual_total = sum(
        written_counts.values()
    )

    conservation = (
        expected_total
        == actual_total
    )

    if not conservation:

        raise RuntimeError(
            f"Final row conservation "
            f"FAILED for {dataset_name}: "
            f"{expected_total} != "
            f"{actual_total}"
        )

    print()
    print(
        f"{dataset_name} "
        f"row conservation: PASS"
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = {

        "dataset": dataset_name,

        "status": "PASS",

        "source_root": str(
            dataset_path
        ),

        "source_files": [
            file.name
            for file in csv_files
        ],

        "feature_source":
            "data/features",

        "split_method":
            "duplicate_aware_stratified_group_split",

        "train_ratio":
            CIC_TRAIN_RATIO,

        "validation_ratio":
            CIC_VAL_RATIO,

        "test_ratio":
            CIC_TEST_RATIO,

        "duplicate_policy":
            (
                "Identical feature fingerprints "
                "remain in the same split."
            ),

        "target_columns_excluded":
            sorted(TARGET_COLUMNS),

        "feature_count":
            len(
                result["feature_columns"]
            ),

        "total_rows":
            expected_total,

        "unique_feature_groups":
            result["unique_groups"],

        "duplicate_rows":
            result["duplicate_rows"],

        "split_counts":
            written_counts,

        "row_conservation": {

            "expected":
                expected_total,

            "written":
                actual_total,

            "passed":
                conservation,
        },

        "output_root":
            str(output_dir),
    }

    save_manifest(
        dataset_name,
        manifest
    )


# ============================================================
# PROCESS UNSW-NB15
# ============================================================

def process_unsw_dataset(
    dataset_name,
    dataset_path
):
    """
    Process UNSW-NB15.

    Official training:
        85% train
        15% validation

    Official testing:
        100% test
    """

    csv_files = get_csv_files(
        dataset_path
    )

    training_file = None
    testing_file = None

    # --------------------------------------------------------
    # Identify official files.
    # --------------------------------------------------------

    for csv_file in csv_files:

        name = csv_file.name.lower()

        if "training-set" in name:

            training_file = csv_file

        elif "testing-set" in name:

            testing_file = csv_file

    if training_file is None:

        raise FileNotFoundError(
            "Could not find "
            "UNSW_NB15_training-set.csv"
        )

    if testing_file is None:

        raise FileNotFoundError(
            "Could not find "
            "UNSW_NB15_testing-set.csv"
        )

    print()
    print("=" * 70)
    print(
        "UNSW-NB15 OFFICIAL SPLIT"
    )
    print("=" * 70)

    print()
    print(
        "Official training:"
    )

    print(
        f"  {training_file}"
    )

    print()
    print(
        "Official testing:"
    )

    print(
        f"  {testing_file}"
    )

    # --------------------------------------------------------
    # Build only TRAIN/VALIDATION assignment
    # using the official training file.
    # --------------------------------------------------------

    train_val_ratios = {

        "train":
            UNSW_TRAIN_RATIO,

        "validation":
            UNSW_VAL_RATIO,
    }

    result = build_group_assignment(
        [training_file],
        train_val_ratios,
        "unswnb15_official_training"
    )

    # --------------------------------------------------------
    # Final output directory
    # --------------------------------------------------------

    output_dir = (
        OUTPUT_ROOT
        / dataset_name
    )

    safe_remove_directory(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Temporary train/validation directory
    # --------------------------------------------------------

    temp_output = (
        OUTPUT_ROOT
        / "_unswnb15_train_validation"
    )

    safe_remove_directory(
        temp_output
    )

    temp_output.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Write official training -> train/validation
    # --------------------------------------------------------

    train_val_counts = (
        write_group_assigned_splits(
            [training_file],
            result["assignment"],
            result["feature_columns"],
            temp_output,
            splits=[
                "train",
                "validation"
            ]
        )
    )

    # --------------------------------------------------------
    # Create final train/validation directories.
    # --------------------------------------------------------

    for split in [
        "train",
        "validation"
    ]:

        destination_dir = (
            output_dir / split
        )

        destination_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        source_dir = (
            temp_output / split
        )

        for source_file in sorted(
            source_dir.glob("*.csv")
        ):

            shutil.move(
                str(source_file),
                str(
                    destination_dir
                    / source_file.name
                )
            )

    # --------------------------------------------------------
    # Remove temporary directory.
    # --------------------------------------------------------

    safe_remove_directory(
        temp_output
    )

    # --------------------------------------------------------
    # Preserve official test set.
    # --------------------------------------------------------

    official_test_rows = (
        write_official_test_file(
            testing_file,
            output_dir
        )
    )

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    official_training_rows = (
        result["total_rows"]
    )

    total_rows = (
        official_training_rows
        + official_test_rows
    )

    final_counts = {

        "train":
            train_val_counts["train"],

        "validation":
            train_val_counts["validation"],

        "test":
            official_test_rows,
    }

    final_written = sum(
        final_counts.values()
    )

    conservation = (
        final_written
        == total_rows
    )

    if not conservation:

        raise RuntimeError(
            "UNSW row conservation "
            "FAILED: "
            f"{final_written} != "
            f"{total_rows}"
        )

    print()
    print(
        "UNSW-NB15 final split counts:"
    )

    for split, count in (
        final_counts.items()
    ):

        percentage = (
            count / total_rows * 100
        )

        print(
            f"  {split:12s}: "
            f"{count:,} "
            f"({percentage:.6f}%)"
        )

    print()
    print(
        "UNSW-NB15 "
        "row conservation: PASS"
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = {

        "dataset":
            dataset_name,

        "status":
            "PASS",

        "source_root":
            str(dataset_path),

        "source_files": [
            training_file.name,
            testing_file.name,
        ],

        "feature_source":
            "data/features",

        "split_method":
            (
                "official_unsw_train_test_"
                "boundary_preserved"
            ),

        "official_training_file":
            training_file.name,

        "official_testing_file":
            testing_file.name,

        "official_test_preserved":
            True,

        "official_training_split": {

            "train_ratio":
                UNSW_TRAIN_RATIO,

            "validation_ratio":
                UNSW_VAL_RATIO,

            "test_ratio":
                0.0,

            "method":
                "duplicate_aware_group_split",
        },

        "official_test_split": {

            "train_ratio":
                0.0,

            "validation_ratio":
                0.0,

            "test_ratio":
                1.0,

            "method":
                "official_test_set_preserved",
        },

        "duplicate_policy":
            (
                "Feature-identical groups "
                "inside the official training "
                "file remain in the same "
                "train/validation split."
            ),

        "official_test_policy":
            (
                "The official UNSW-NB15 "
                "testing file is kept entirely "
                "as test data."
            ),

        "target_columns_excluded":
            sorted(TARGET_COLUMNS),

        "feature_count":
            len(
                result["feature_columns"]
            ),

        "official_training_rows":
            official_training_rows,

        "official_test_rows":
            official_test_rows,

        "total_rows":
            total_rows,

        "split_counts":
            final_counts,

        "row_conservation": {

            "expected":
                total_rows,

            "written":
                final_written,

            "passed":
                conservation,
        },

        "output_root":
            str(output_dir),

        "methodological_note":
            (
                "The official UNSW-NB15 "
                "training/testing boundary "
                "is intentionally preserved. "
                "Feature-identical rows may "
                "exist across the official "
                "training and testing files; "
                "they are not moved across "
                "the official boundary."
            ),
    }

    save_manifest(
        dataset_name,
        manifest
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "CyberSecAI Training Split Builder"
    )
    print("=" * 70)

    print()
    print(
        "Project root:"
    )

    print(
        f"  {ROOT}"
    )

    print()
    print(
        "Feature root:"
    )

    print(
        f"  {FEATURE_ROOT}"
    )

    print()
    print(
        "Output root:"
    )

    print(
        f"  {OUTPUT_ROOT}"
    )

    # --------------------------------------------------------
    # Validate feature directories first.
    # --------------------------------------------------------

    print()
    print(
        "Checking feature directories..."
    )

    for dataset_name, path in (
        DATASETS.items()
    ):

        if not path.exists():

            raise FileNotFoundError(
                f"Missing feature directory "
                f"for {dataset_name}: "
                f"{path}"
            )

        csv_count = len(
            get_csv_files(path)
        )

        print(
            f"  {dataset_name:12s}: "
            f"{csv_count} CSV files"
        )

    # --------------------------------------------------------
    # Create metadata/output roots.
    # --------------------------------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    METADATA_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # CIC-IDS2017
    # ========================================================

    process_cic_dataset(
        "cicids2017",
        DATASETS["cicids2017"]
    )

    # ========================================================
    # CIC-IDS2018
    # ========================================================

    process_cic_dataset(
        "cicids2018",
        DATASETS["cicids2018"]
    )

    # ========================================================
    # UNSW-NB15
    # ========================================================

    process_unsw_dataset(
        "unswnb15",
        DATASETS["unswnb15"]
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print(
        "ALL TRAINING SPLITS COMPLETED"
    )
    print("=" * 70)

    print()
    print(
        "Generated datasets:"
    )

    print(
        f"  CIC-IDS2017 : "
        f"{OUTPUT_ROOT / 'cicids2017'}"
    )

    print(
        f"  CIC-IDS2018 : "
        f"{OUTPUT_ROOT / 'cicids2018'}"
    )

    print(
        f"  UNSW-NB15   : "
        f"{OUTPUT_ROOT / 'unswnb15'}"
    )

    print()
    print(
        "Generated manifests:"
    )

    print(
        f"  {METADATA_ROOT / 'cicids2017_training_split_manifest.json'}"
    )

    print(
        f"  {METADATA_ROOT / 'cicids2018_training_split_manifest.json'}"
    )

    print(
        f"  {METADATA_ROOT / 'unswnb15_training_split_manifest.json'}"
    )

    print()
    print(
        "Next step: validate the generated "
        "split files."
    )


if __name__ == "__main__":
    main()  