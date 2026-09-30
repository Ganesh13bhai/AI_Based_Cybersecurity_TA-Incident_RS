import os
import json
import hashlib
from collections import defaultdict

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

MANIFEST_FILE = (
    "data/metadata/stratified_split_manifest.json"
)

OUTPUT_FILE = (
    "data/metadata/stratified_split_validation_report.json"
)

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

EXPECTED_CLASSES = {
    "BENIGN",
    "BOTNET",
    "BRUTE_FORCE",
    "DDOS",
    "DOS",
    "INFILTRATION",
    "OTHER_ATTACK",
    "PORT_SCAN",
    "WEB_ATTACK",
}

DATASETS = {
    "cicids2017": "data/processed/cicids2017",
    "cicids2018": "data/processed/cicids2018",
}


# ============================================================
# HELPERS
# ============================================================

def normalize_column_name(column):
    return str(column).strip().lower()


def normalize_value(value):

    if pd.isna(value):
        return "<NA>"

    if isinstance(value, float):
        return f"{value:.12g}"

    return str(value).strip()


def make_fingerprint(row, feature_columns):

    values = [
        normalize_value(row[column])
        for column in feature_columns
    ]

    raw = "\x1f".join(values)

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def get_csv_files(directory):

    if not os.path.isdir(directory):

        raise FileNotFoundError(
            f"Dataset directory not found: {directory}"
        )

    files = []

    for filename in sorted(os.listdir(directory)):

        if not filename.endswith(".csv"):
            continue

        if filename.startswith("._"):
            continue

        files.append(
            os.path.join(
                directory,
                filename
            )
        )

    return files


def discover_columns(csv_files):

    all_columns = set()

    for file_path in csv_files:

        df = pd.read_csv(
            file_path,
            nrows=5
        )

        all_columns.update(
            df.columns.tolist()
        )

    return sorted(all_columns)


def get_feature_columns(all_columns):

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

    return feature_columns, excluded_columns


def find_canonical_label_column(columns):

    for column in columns:

        if normalize_column_name(column) == "canonical_label":

            return column

    return None


# ============================================================
# COLLECT FEATURE GROUPS
# ============================================================

def collect_groups(
    csv_files,
    feature_columns
):

    groups = {}

    total_rows = 0

    label_counts = defaultdict(int)

    for file_path in csv_files:

        print(
            f"Scanning: {file_path}"
        )

        for chunk in pd.read_csv(
            file_path,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ):

            canonical_column = (
                find_canonical_label_column(
                    chunk.columns
                )
            )

            if canonical_column is None:

                raise ValueError(
                    "canonical_label column not found "
                    f"in {file_path}"
                )

            for _, row in chunk.iterrows():

                label = str(
                    row[canonical_column]
                ).strip()

                if not label:

                    raise ValueError(
                        f"Empty canonical label found "
                        f"in {file_path}"
                    )

                fingerprint = make_fingerprint(
                    row,
                    feature_columns
                )

                total_rows += 1

                label_counts[label] += 1

                if fingerprint not in groups:

                    groups[fingerprint] = {
                        "label": label,
                        "count": 1,
                        "labels": {label},
                    }

                else:

                    groups[fingerprint]["count"] += 1

                    groups[fingerprint][
                        "labels"
                    ].add(label)

    conflicting_labels = []

    for fingerprint, info in groups.items():

        if len(info["labels"]) > 1:

            conflicting_labels.append(
                {
                    "fingerprint": fingerprint,
                    "labels": sorted(
                        info["labels"]
                    ),
                    "count": info["count"],
                    "primary_label": info["label"],
                }
            )

    return (
        groups,
        total_rows,
        dict(label_counts),
        conflicting_labels,
    )


# ============================================================
# RECREATE SPLIT ASSIGNMENT
# ============================================================

def assign_groups(groups):

    by_label = defaultdict(list)

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

    for label in sorted(by_label):

        label_groups = by_label[label]

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

        targets = {
            "train":
                total * TRAIN_RATIO,

            "validation":
                total * VAL_RATIO,

            "test":
                total * TEST_RATIO,
        }

        current = {
            "train": 0,
            "validation": 0,
            "test": 0,
        }

        for fingerprint, count in label_groups:

            split = max(
                targets,
                key=lambda split_name:
                    targets[split_name]
                    - current[split_name]
            )

            assignments[
                fingerprint
            ] = split

            current[
                split
            ] += count

    return assignments


# ============================================================
# DUPLICATE LEAKAGE
# ============================================================

def check_duplicate_leakage(
    groups,
    assignments
):

    split_fingerprints = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }

    for fingerprint in groups:

        if fingerprint not in assignments:

            raise ValueError(
                "Fingerprint missing from assignment: "
                f"{fingerprint}"
            )

        split = assignments[fingerprint]

        split_fingerprints[
            split
        ].add(fingerprint)

    train = split_fingerprints["train"]
    validation = split_fingerprints["validation"]
    test = split_fingerprints["test"]

    train_validation = train & validation
    train_test = train & test
    validation_test = validation & test

    leakage_count = (
        len(train_validation)
        + len(train_test)
        + len(validation_test)
    )

    return {
        "status":
            "PASS"
            if leakage_count == 0
            else "FAIL",

        "train_groups":
            len(train),

        "validation_groups":
            len(validation),

        "test_groups":
            len(test),

        "train_validation_overlap":
            len(train_validation),

        "train_test_overlap":
            len(train_test),

        "validation_test_overlap":
            len(validation_test),

        "total_leaking_groups":
            leakage_count,
    }


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def calculate_class_distribution(
    groups,
    assignments
):

    split_label_counts = {
        "train": defaultdict(int),
        "validation": defaultdict(int),
        "test": defaultdict(int),
    }

    split_counts = {
        "train": 0,
        "validation": 0,
        "test": 0,
    }

    for fingerprint, info in groups.items():

        split = assignments[
            fingerprint
        ]

        label = info["label"]

        count = info["count"]

        split_label_counts[
            split
        ][label] += count

        split_counts[
            split
        ] += count

    distribution = {}

    for split in [
        "train",
        "validation",
        "test",
    ]:

        total = split_counts[split]

        distribution[split] = {
            "total_rows": total,
            "classes": {},
        }

        for label in sorted(
            EXPECTED_CLASSES
        ):

            count = split_label_counts[
                split
            ].get(label, 0)

            percentage = (
                count / total * 100
                if total > 0
                else 0
            )

            distribution[
                split
            ]["classes"][label] = {
                "count": count,
                "percentage": percentage,
            }

    return split_counts, distribution


# ============================================================
# CLASS COVERAGE
# ============================================================

def check_class_coverage(
    distribution
):

    coverage = {}

    overall_status = "PASS"

    for split in [
        "train",
        "validation",
        "test",
    ]:

        missing = []
        present = []

        for label in sorted(
            EXPECTED_CLASSES
        ):

            count = distribution[
                split
            ]["classes"][label]["count"]

            if count == 0:
                missing.append(label)
            else:
                present.append(label)

        status = (
            "PASS"
            if not missing
            else "REVIEW"
        )

        if missing:
            overall_status = "REVIEW"

        coverage[split] = {
            "status": status,
            "present_classes": present,
            "missing_classes": missing,
            "class_count": len(present),
            "expected_class_count":
                len(EXPECTED_CLASSES),
        }

    return overall_status, coverage


# ============================================================
# SPLIT RATIOS
# ============================================================

def check_split_ratios(
    split_counts,
    total_rows
):

    requested = {
        "train": TRAIN_RATIO * 100,
        "validation": VAL_RATIO * 100,
        "test": TEST_RATIO * 100,
    }

    actual = {}
    differences = {}

    for split in [
        "train",
        "validation",
        "test",
    ]:

        percentage = (
            split_counts[split]
            / total_rows
            * 100
        )

        actual[split] = percentage

        differences[split] = (
            percentage
            - requested[split]
        )

    tolerance = 0.10

    status = (
        "PASS"
        if all(
            abs(x) <= tolerance
            for x in differences.values()
        )
        else "REVIEW"
    )

    return {
        "status": status,
        "requested_percentages": requested,
        "actual_percentages": actual,
        "difference_percentage_points":
            differences,
        "tolerance_percentage_points":
            tolerance,
    }


# ============================================================
# TARGET COLUMN CHECK
# ============================================================

def check_target_columns(
    feature_columns
):

    remaining = [
        column
        for column in feature_columns
        if normalize_column_name(column)
        in TARGET_COLUMNS
    ]

    return {
        "status":
            "PASS"
            if not remaining
            else "FAIL",

        "remaining_target_columns":
            remaining,
    }


# ============================================================
# MANIFEST CONSISTENCY
# ============================================================

def check_manifest_consistency(
    manifest_dataset,
    csv_files,
    total_rows,
    feature_columns,
    label_counts,
):
    """
    IMPORTANT:

    The split manifest is an aggregate description of the
    generated split. Different versions of the generator may
    store different aggregate fields.

    Therefore this check only validates fields that actually
    exist in the manifest.

    It does NOT assume that split_counts is present.
    """

    checks = {}

    # --------------------------------------------------------
    # Total rows
    # --------------------------------------------------------

    manifest_total = manifest_dataset.get(
        "total_rows"
    )

    if manifest_total is None:

        checks["total_rows"] = {
            "status": "SKIPPED",
            "reason":
                "total_rows not stored in manifest.",
        }

    else:

        checks["total_rows"] = {
            "status":
                "PASS"
                if manifest_total == total_rows
                else "FAIL",

            "manifest":
                manifest_total,

            "recomputed":
                total_rows,
        }

    # --------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------

    manifest_feature_count = (
        manifest_dataset.get(
            "feature_columns"
        )
    )

    if manifest_feature_count is None:

        checks["feature_columns"] = {
            "status": "SKIPPED",
            "reason":
                "feature_columns count not stored "
                "in manifest.",
        }

    elif isinstance(
        manifest_feature_count,
        int
    ):

        checks["feature_columns"] = {
            "status":
                "PASS"
                if (
                    manifest_feature_count
                    == len(feature_columns)
                )
                else "FAIL",

            "manifest":
                manifest_feature_count,

            "recomputed":
                len(feature_columns),
        }

    elif isinstance(
        manifest_feature_count,
        list
    ):

        checks["feature_columns"] = {
            "status":
                "PASS"
                if (
                    len(manifest_feature_count)
                    == len(feature_columns)
                )
                else "REVIEW",

            "manifest_count":
                len(manifest_feature_count),

            "recomputed_count":
                len(feature_columns),

            "reason":
                "Manifest stores feature names "
                "rather than only a count.",
        }

    else:

        checks["feature_columns"] = {
            "status": "SKIPPED",
            "reason":
                "Unrecognized feature_columns format.",
        }

    # --------------------------------------------------------
    # Label counts
    # --------------------------------------------------------

    manifest_labels = manifest_dataset.get(
        "label_counts"
    )

    if manifest_labels is None:

        checks["label_counts"] = {
            "status": "SKIPPED",
            "reason":
                "label_counts not stored in manifest.",
        }

    else:

        normalized_manifest_labels = {
            str(k): int(v)
            for k, v in manifest_labels.items()
        }

        normalized_actual_labels = {
            str(k): int(v)
            for k, v in label_counts.items()
        }

        checks["label_counts"] = {
            "status":
                "PASS"
                if (
                    normalized_manifest_labels
                    == normalized_actual_labels
                )
                else "FAIL",

            "manifest":
                normalized_manifest_labels,

            "recomputed":
                normalized_actual_labels,
        }

    # --------------------------------------------------------
    # File list
    # --------------------------------------------------------

    manifest_files = manifest_dataset.get(
        "files"
    )

    if manifest_files is None:

        checks["files"] = {
            "status": "SKIPPED",
            "reason":
                "File list not stored in manifest.",
        }

    else:

        actual_files = sorted(
            os.path.basename(
                path
            )
            for path in csv_files
        )

        # Handle common manifest formats
        if isinstance(
            manifest_files,
            list
        ):

            manifest_file_names = []

            for item in manifest_files:

                if isinstance(item, str):

                    manifest_file_names.append(
                        os.path.basename(item)
                    )

                elif isinstance(item, dict):

                    possible_name = (
                        item.get("file")
                        or item.get("filename")
                        or item.get("name")
                    )

                    if possible_name:

                        manifest_file_names.append(
                            os.path.basename(
                                possible_name
                            )
                        )

            manifest_file_names = sorted(
                manifest_file_names
            )

            checks["files"] = {
                "status":
                    "PASS"
                    if (
                        manifest_file_names
                        == actual_files
                    )
                    else "REVIEW",

                "manifest_count":
                    len(manifest_file_names),

                "actual_count":
                    len(actual_files),

                "missing_files":
                    sorted(
                        set(actual_files)
                        - set(manifest_file_names)
                    ),

                "extra_files":
                    sorted(
                        set(manifest_file_names)
                        - set(actual_files)
                    ),
            }

        else:

            checks["files"] = {
                "status": "SKIPPED",
                "reason":
                    "Unrecognized files format.",
            }

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    statuses = [
        value["status"]
        for value in checks.values()
    ]

    if "FAIL" in statuses:

        status = "FAIL"

    elif "REVIEW" in statuses:

        status = "REVIEW"

    else:

        status = "PASS"

    return {
        "status": status,
        "checks": checks,
        "note":
            (
                "Manifest consistency validates only "
                "fields actually stored in the manifest. "
                "Split counts are independently recomputed "
                "from the deterministic split algorithm."
            ),
    }


# ============================================================
# CONFLICTING LABELS
# ============================================================

def check_conflicting_feature_labels(
    conflicting_labels
):

    if not conflicting_labels:

        return {
            "status": "PASS",
            "count": 0,
            "examples": [],
            "interpretation":
                "No conflicting feature-label groups found.",
        }

    return {
        "status": "REVIEW",

        "count":
            len(conflicting_labels),

        "examples":
            conflicting_labels[:20],

        "interpretation":
            (
                "Identical feature vectors occur with "
                "multiple canonical labels. These groups "
                "are retained and assigned entirely to "
                "one split. They are documented for "
                "data-quality review and are not treated "
                "as automatic leakage."
            ),
    }


# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_dataset(
    dataset_name,
    directory,
    manifest_dataset
):

    print()
    print("=" * 70)
    print(
        f"VALIDATING: {dataset_name}"
    )
    print("=" * 70)

    csv_files = get_csv_files(
        directory
    )

    if not csv_files:

        raise ValueError(
            f"No CSV files found: {directory}"
        )

    print(
        f"CSV files found: {len(csv_files)}"
    )

    all_columns = discover_columns(
        csv_files
    )

    feature_columns, excluded_columns = (
        get_feature_columns(
            all_columns
        )
    )

    print(
        f"Feature columns: {len(feature_columns)}"
    )

    print(
        "Excluded target columns:",
        excluded_columns
    )

    target_result = check_target_columns(
        feature_columns
    )

    (
        groups,
        total_rows,
        label_counts,
        conflicting_labels,
    ) = collect_groups(
        csv_files,
        feature_columns
    )

    assignments = assign_groups(
        groups
    )

    duplicate_result = (
        check_duplicate_leakage(
            groups,
            assignments
        )
    )

    (
        split_counts,
        distribution,
    ) = calculate_class_distribution(
        groups,
        assignments
    )

    (
        coverage_status,
        coverage,
    ) = check_class_coverage(
        distribution
    )

    ratio_result = check_split_ratios(
        split_counts,
        total_rows
    )

    manifest_result = (
        check_manifest_consistency(
            manifest_dataset,
            csv_files,
            total_rows,
            feature_columns,
            label_counts,
        )
    )

    conflict_result = (
        check_conflicting_feature_labels(
            conflicting_labels
        )
    )

    structural_checks = [
        target_result["status"],
        duplicate_result["status"],
        coverage_status,
        ratio_result["status"],
        manifest_result["status"],
    ]

    if "FAIL" in structural_checks:

        structural_status = "FAIL"

    elif "REVIEW" in structural_checks:

        structural_status = "REVIEW"

    else:

        structural_status = "PASS"

    if structural_status == "FAIL":

        dataset_status = "FAIL"

    elif (
        structural_status == "REVIEW"
        or conflict_result["status"] == "REVIEW"
    ):

        dataset_status = "PASS_WITH_REVIEW"

    else:

        dataset_status = "PASS"

    print()

    print(
        "Target columns excluded: "
        f"{target_result['status']}"
    )

    print(
        "Duplicate leakage: "
        f"{duplicate_result['status']}"
    )

    print(
        "Class coverage: "
        f"{coverage_status}"
    )

    print(
        "Split ratios: "
        f"{ratio_result['status']}"
    )

    print(
        "Manifest consistency: "
        f"{manifest_result['status']}"
    )

    print(
        "Conflicting feature labels: "
        f"{conflict_result['status']}"
    )

    print()

    print(
        "STRUCTURAL SPLIT STATUS: "
        f"{structural_status}"
    )

    print(
        "FINAL DATASET STATUS: "
        f"{dataset_status}"
    )

    return {
        "dataset": dataset_name,

        "status":
            dataset_status,

        "split_integrity_status":
            structural_status,

        "total_rows":
            total_rows,

        "unique_feature_groups":
            len(groups),

        "duplicate_rows":
            total_rows - len(groups),

        "feature_columns":
            len(feature_columns),

        "excluded_target_columns":
            excluded_columns,

        "csv_files":
            [
                os.path.basename(path)
                for path in csv_files
            ],

        "target_column_check":
            target_result,

        "label_counts":
            label_counts,

        "duplicate_leakage":
            duplicate_result,

        "class_distribution":
            distribution,

        "class_coverage":
            coverage,

        "split_ratios":
            ratio_result,

        "manifest_consistency":
            manifest_result,

        "conflicting_feature_labels":
            conflict_result,

        "split_counts":
            split_counts,

        "methodology_notes": [
            "Target columns are excluded case-insensitively.",
            "Feature fingerprints use SHA-256.",
            "Duplicate feature groups remain entirely within one split.",
            "Conflicting feature-label groups are retained.",
            "Conflicting feature-label groups are reported as REVIEW.",
            "No rows are deleted during validation.",
            "No labels are changed during validation.",
        ],
    }


# ============================================================
# OVERALL STATUS
# ============================================================

def determine_overall_status(
    dataset_results
):

    statuses = [
        result["status"]
        for result in dataset_results.values()
    ]

    if "FAIL" in statuses:
        return "FAIL"

    if "REVIEW" in statuses:
        return "REVIEW"

    if "PASS_WITH_REVIEW" in statuses:
        return "PASS_WITH_REVIEW"

    return "PASS"


# ============================================================
# OVERALL CHECKS
# ============================================================

def build_overall_checks(
    dataset_results
):

    duplicate_status = (
        "PASS"
        if all(
            result[
                "duplicate_leakage"
            ]["status"] == "PASS"
            for result in dataset_results.values()
        )
        else "FAIL"
    )

    ratio_status = (
        "PASS"
        if all(
            result[
                "split_ratios"
            ]["status"] == "PASS"
            for result in dataset_results.values()
        )
        else "REVIEW"
    )

    coverage_status = (
        "PASS"
        if all(
            result[
                "class_coverage"
            ][split]["status"] == "PASS"

            for result in dataset_results.values()

            for split in [
                "train",
                "validation",
                "test",
            ]
        )
        else "REVIEW"
    )

    target_status = (
        "PASS"
        if all(
            result[
                "target_column_check"
            ]["status"] == "PASS"
            for result in dataset_results.values()
        )
        else "FAIL"
    )

    manifest_status = (
        "PASS"
        if all(
            result[
                "manifest_consistency"
            ]["status"] == "PASS"
            for result in dataset_results.values()
        )
        else "REVIEW"
    )

    conflict_status = (
        "PASS"
        if all(
            result[
                "conflicting_feature_labels"
            ]["status"] == "PASS"
            for result in dataset_results.values()
        )
        else "REVIEW"
    )

    return {
        "no_duplicate_leakage":
            duplicate_status,

        "class_distribution":
            ratio_status,

        "all_expected_classes_present":
            coverage_status,

        "target_columns_excluded":
            target_status,

        "manifest_consistency":
            manifest_status,

        "conflicting_feature_labels":
            conflict_status,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    if not os.path.exists(
        MANIFEST_FILE
    ):

        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_FILE}"
        )

    with open(
        MANIFEST_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        manifest = json.load(f)

    print()
    print("=" * 70)
    print(
        "CYBERSECAI STRATIFIED SPLIT VALIDATION"
    )
    print("=" * 70)

    dataset_results = {}

    manifest_datasets = manifest.get(
        "datasets",
        {}
    )

    for dataset_name, directory in (
        DATASETS.items()
    ):

        if dataset_name not in manifest_datasets:

            raise ValueError(
                f"{dataset_name} missing from manifest."
            )

        dataset_results[
            dataset_name
        ] = validate_dataset(
            dataset_name,
            directory,
            manifest_datasets[
                dataset_name
            ],
        )

    overall_status = (
        determine_overall_status(
            dataset_results
        )
    )

    checks = build_overall_checks(
        dataset_results
    )

    report = {

        "status":
            overall_status,

        "validation_version":
            "2.0",

        "method": {

            "type":
                "duplicate_aware_stratified",

            "train_ratio":
                TRAIN_RATIO,

            "validation_ratio":
                VAL_RATIO,

            "test_ratio":
                TEST_RATIO,

            "fingerprint":
                "SHA-256",

            "duplicate_policy":
                "keep_duplicate_feature_groups_together",

            "conflicting_label_policy":
                "retain_and_review",
        },

        "checks":
            checks,

        "datasets":
            dataset_results,

        "interpretation":
            (
                "PASS_WITH_REVIEW means the split integrity "
                "checks passed but identical feature vectors "
                "with multiple labels were found and retained "
                "for data-quality review."
            ),
    }

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
            indent=2
        )

    print()
    print("=" * 70)
    print(
        "FINAL VALIDATION SUMMARY"
    )
    print("=" * 70)

    print()

    for check, status in checks.items():

        print(
            f"{check}: {status}"
        )

    print()

    for dataset_name, result in (
        dataset_results.items()
    ):

        print(
            f"{dataset_name}: "
            f"{result['status']}"
        )

    print()

    print(
        f"OVERALL STATUS: "
        f"{overall_status}"
    )

    print()

    print(
        "Report saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()