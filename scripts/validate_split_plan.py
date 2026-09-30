import json
import glob
import os

import pandas as pd


# =========================================================
# Configuration
# =========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PROCESSED_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

SPLIT_PLAN_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "metadata",
    "split_plan.json"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "metadata",
    "split_validation_report.json"
)

CHUNK_SIZE = 100_000

LABEL_COLUMN = "canonical_label"


# =========================================================
# Helpers
# =========================================================

def find_file(dataset, filename):

    dataset_dir_map = {
        "CIC-IDS2017": "cicids2017",
        "CIC-IDS2018": "cicids2018",
        "UNSW-NB15": "unswnb15"
    }

    directory = dataset_dir_map.get(dataset)

    if directory is None:
        return None

    path = os.path.join(
        PROCESSED_DIR,
        directory,
        filename
    )

    if os.path.exists(path):
        return path

    return None


def count_file_rows(file):

    total = 0

    for chunk in pd.read_csv(
        file,
        usecols=[LABEL_COLUMN],
        chunksize=CHUNK_SIZE
    ):
        total += len(chunk)

    return total


def get_label_counts(file):

    counts = {}

    for chunk in pd.read_csv(
        file,
        usecols=[LABEL_COLUMN],
        chunksize=CHUNK_SIZE
    ):

        values = (
            chunk[LABEL_COLUMN]
            .value_counts()
        )

        for label, count in values.items():

            label = str(label)

            counts[label] = (
                counts.get(label, 0)
                + int(count)
            )

    return counts


def merge_counts(destination, source):

    for label, count in source.items():

        destination[label] = (
            destination.get(label, 0)
            + count
        )


def calculate_percentages(counts):

    total = sum(counts.values())

    if total == 0:
        return {}

    return {
        label: round(
            (count / total) * 100,
            4
        )
        for label, count in counts.items()
    }


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 70)
    print("CyberSecAI - Step 9 Split Validation")
    print("=" * 70)

    if not os.path.exists(SPLIT_PLAN_FILE):

        raise FileNotFoundError(
            f"Split plan not found:\n"
            f"{SPLIT_PLAN_FILE}"
        )

    with open(
        SPLIT_PLAN_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        split_plan = json.load(f)


    report = {

        "version": "1.0",

        "description": (
            "Validation report for the "
            "CyberSecAI leakage-aware split plan."
        ),

        "target_ratio": split_plan[
            "target_ratio"
        ],

        "datasets": {},

        "overall": {
            "status": "PASS",
            "issues": []
        }
    }


    # =====================================================
    # Process each dataset
    # =====================================================

    for dataset, items in (
        split_plan["datasets"].items()
    ):

        print("\n" + "=" * 70)
        print(dataset)
        print("=" * 70)


        dataset_report = {

            "files": [],

            "splits": {

                "train": {
                    "rows": 0,
                    "labels": {}
                },

                "validation": {
                    "rows": 0,
                    "labels": {}
                },

                "test": {
                    "rows": 0,
                    "labels": {}
                },

                "train_validation_source": {
                    "rows": 0,
                    "labels": {}
                }
            },

            "status": "PASS"
        }


        # -------------------------------------------------
        # Process files
        # -------------------------------------------------

        for item in items:

            filename = item["file"]
            group = item["group"]
            split = item["split"]

            file = find_file(
                dataset,
                filename
            )

            if file is None:

                message = (
                    f"Missing file: "
                    f"{dataset}/{filename}"
                )

                report["overall"]["issues"].append(
                    message
                )

                dataset_report["status"] = "ERROR"

                continue


            print(
                f"\nProcessing: {filename}"
            )

            print(
                f"Group     : {group}"
            )

            print(
                f"Split     : {split}"
            )


            # ---------------------------------------------
            # Count rows
            # ---------------------------------------------

            rows = count_file_rows(file)

            labels = get_label_counts(file)


            print(
                f"Rows      : {rows:,}"
            )


            print(
                "Labels    : "
                + ", ".join(
                    f"{k}={v:,}"
                    for k, v in sorted(
                        labels.items()
                    )
                )
            )


            dataset_report["files"].append(
                {
                    "file": filename,
                    "group": group,
                    "split": split,
                    "rows": rows,
                    "labels": labels
                }
            )


            # ---------------------------------------------
            # Add to split totals
            # ---------------------------------------------

            if split in dataset_report["splits"]:

                dataset_report[
                    "splits"
                ][split]["rows"] += rows

                merge_counts(
                    dataset_report[
                        "splits"
                    ][split]["labels"],
                    labels
                )


        # =================================================
        # Dataset split statistics
        # =================================================

        print("\n" + "-" * 70)
        print("Split statistics")
        print("-" * 70)


        # For CIC datasets:
        #
        # train + validation + test
        #
        # For UNSW:
        #
        # train_validation_source + test
        #

        if dataset in {
            "CIC-IDS2017",
            "CIC-IDS2018"
        }:

            train_rows = dataset_report[
                "splits"
            ]["train"]["rows"]

            validation_rows = dataset_report[
                "splits"
            ]["validation"]["rows"]

            test_rows = dataset_report[
                "splits"
            ]["test"]["rows"]

            total_rows = (
                train_rows
                + validation_rows
                + test_rows
            )


            if total_rows > 0:

                train_pct = (
                    train_rows
                    / total_rows
                    * 100
                )

                validation_pct = (
                    validation_rows
                    / total_rows
                    * 100
                )

                test_pct = (
                    test_rows
                    / total_rows
                    * 100
                )

            else:

                train_pct = 0
                validation_pct = 0
                test_pct = 0


            dataset_report[
                "row_distribution"
            ] = {

                "total_rows": total_rows,

                "train": {
                    "rows": train_rows,
                    "percentage": round(
                        train_pct,
                        4
                    )
                },

                "validation": {
                    "rows": validation_rows,
                    "percentage": round(
                        validation_pct,
                        4
                    )
                },

                "test": {
                    "rows": test_rows,
                    "percentage": round(
                        test_pct,
                        4
                    )
                }
            }


            print(
                f"Train      : "
                f"{train_rows:,} "
                f"({train_pct:.2f}%)"
            )

            print(
                f"Validation : "
                f"{validation_rows:,} "
                f"({validation_pct:.2f}%)"
            )

            print(
                f"Test       : "
                f"{test_rows:,} "
                f"({test_pct:.2f}%)"
            )


            # ---------------------------------------------
            # Check class coverage
            # ---------------------------------------------

            expected_labels = set()

            for split_name in [
                "train",
                "validation",
                "test"
            ]:

                expected_labels.update(
                    dataset_report[
                        "splits"
                    ][split_name][
                        "labels"
                    ].keys()
                )


            missing_classes = {}

            for split_name in [
                "train",
                "validation",
                "test"
            ]:

                present = set(
                    dataset_report[
                        "splits"
                    ][split_name][
                        "labels"
                    ].keys()
                )

                missing = sorted(
                    expected_labels - present
                )

                missing_classes[
                    split_name
                ] = missing


                if missing:

                    print(
                        f"WARNING: {split_name} "
                        f"missing classes: "
                        f"{missing}"
                    )


            dataset_report[
                "missing_classes"
            ] = missing_classes


            # ---------------------------------------------
            # Split validation status
            # ---------------------------------------------

            if any(
                missing_classes.values()
            ):

                dataset_report[
                    "status"
                ] = "REVIEW"


        # =================================================
        # UNSW validation
        # =================================================

        elif dataset == "UNSW-NB15":

            train_source_rows = (
                dataset_report[
                    "splits"
                ]["train_validation_source"][
                    "rows"
                ]
            )

            test_rows = (
                dataset_report[
                    "splits"
                ]["test"]["rows"]
            )

            total_rows = (
                train_source_rows
                + test_rows
            )


            if total_rows > 0:

                train_source_pct = (
                    train_source_rows
                    / total_rows
                    * 100
                )

                test_pct = (
                    test_rows
                    / total_rows
                    * 100
                )

            else:

                train_source_pct = 0
                test_pct = 0


            dataset_report[
                "row_distribution"
            ] = {

                "total_rows": total_rows,

                "train_validation_source": {
                    "rows": train_source_rows,
                    "percentage": round(
                        train_source_pct,
                        4
                    )
                },

                "official_test": {
                    "rows": test_rows,
                    "percentage": round(
                        test_pct,
                        4
                    )
                }
            }


            print(
                f"Training source : "
                f"{train_source_rows:,} "
                f"({train_source_pct:.2f}%)"
            )

            print(
                f"Official test   : "
                f"{test_rows:,} "
                f"({test_pct:.2f}%)"
            )


            # ---------------------------------------------
            # Check test class coverage
            # ---------------------------------------------

            source_labels = set(
                dataset_report[
                    "splits"
                ]["train_validation_source"][
                    "labels"
                ].keys()
            )

            test_labels = set(
                dataset_report[
                    "splits"
                ]["test"]["labels"].keys()
            )


            missing_test_classes = sorted(
                source_labels - test_labels
            )

            missing_source_classes = sorted(
                test_labels - source_labels
            )


            dataset_report[
                "class_coverage"
            ] = {

                "training_source_labels": sorted(
                    source_labels
                ),

                "test_labels": sorted(
                    test_labels
                ),

                "missing_from_test": (
                    missing_test_classes
                ),

                "missing_from_training_source": (
                    missing_source_classes
                )
            }


            if missing_test_classes:

                print(
                    "WARNING: Classes present in "
                    "training source but absent "
                    "from official test:"
                )

                for label in missing_test_classes:

                    print(
                        f"  - {label}"
                    )

                # This is not automatically an error.
                # Official test distributions are preserved.

                dataset_report[
                    "status"
                ] = "REVIEW"


        report[
            "datasets"
        ][dataset] = dataset_report


    # =====================================================
    # Overall status
    # =====================================================

    statuses = [
        dataset_report["status"]
        for dataset_report
        in report["datasets"].values()
    ]


    if "ERROR" in statuses:

        report["overall"][
            "status"
        ] = "ERROR"

    elif "REVIEW" in statuses:

        report["overall"][
            "status"
        ] = "REVIEW"

    else:

        report["overall"][
            "status"
        ] = "PASS"


    # =====================================================
    # Save report
    # =====================================================

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
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


    # =====================================================
    # Final output
    # =====================================================

    print("\n" + "=" * 70)
    print("FINAL SPLIT VALIDATION")
    print("=" * 70)

    print(
        f"Overall status : "
        f"{report['overall']['status']}"
    )

    print(
        f"Report saved   : "
        f"{OUTPUT_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()