import json
import os


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MANIFEST_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "metadata",
    "split_manifest.json"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "metadata",
    "split_plan.json"
)


# ---------------------------------------------------------
# Explicit group assignments
# ---------------------------------------------------------
#
# We intentionally keep capture groups together.
#
# CIC datasets:
#
# Train:
#   Monday
#   Tuesday
#   Wednesday
#
# Validation:
#   Thursday
#
# Test:
#   Friday
#
# This is a GROUP-AWARE plan, not a row-level 70/15/15 split.
#
# The exact row proportions will be calculated and reported.
#
# UNSW:
#   official_train -> split into train/validation later
#   official_test  -> final test
#

CIC_GROUP_ASSIGNMENTS = {
    "train": [
        "Monday",
        "Tuesday",
        "Wednesday"
    ],
    "validation": [
        "Thursday"
    ],
    "test": [
        "Friday"
    ]
}


UNSW_ASSIGNMENTS = {
    "official_train": "train_validation_source",
    "official_test": "test"
}


def get_split(dataset, group):

    if dataset in {
        "CIC-IDS2017",
        "CIC-IDS2018"
    }:

        for split, groups in CIC_GROUP_ASSIGNMENTS.items():

            if group in groups:
                return split

        return "UNASSIGNED"

    if dataset == "UNSW-NB15":

        return UNSW_ASSIGNMENTS.get(
            group,
            "UNASSIGNED"
        )

    return "UNASSIGNED"


def main():

    print("=" * 70)
    print("CyberSecAI - Step 9 Split Plan Generator")
    print("=" * 70)

    if not os.path.exists(MANIFEST_FILE):
        raise FileNotFoundError(
            f"Split manifest not found:\n{MANIFEST_FILE}"
        )

    with open(
        MANIFEST_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        manifest = json.load(f)

    split_plan = {
        "version": "1.0",

        "description": (
            "Leakage-aware train/validation/test "
            "split plan for CyberSecAI."
        ),

        "strategy": {
            "CIC-IDS2017": {
                "method": "group_based",
                "rule": (
                    "Files from the same capture day "
                    "remain in the same split."
                )
            },

            "CIC-IDS2018": {
                "method": "group_based",
                "rule": (
                    "Files from the same capture day "
                    "remain in the same split."
                )
            },

            "UNSW-NB15": {
                "method": "official_split_preservation",
                "rule": (
                    "Official training and testing "
                    "files remain separated."
                )
            }
        },

        "target_ratio": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15
        },

        "datasets": {},

        "warnings": []
    }

    total_files = 0

    for dataset, items in manifest["datasets"].items():

        split_plan["datasets"][dataset] = []

        print(f"\n{dataset}")
        print("-" * 70)

        for item in items:

            filename = item["file"]
            group = item["group"]

            split = get_split(
                dataset,
                group
            )

            if split == "UNASSIGNED":

                split_plan["warnings"].append(
                    {
                        "dataset": dataset,
                        "file": filename,
                        "group": group,
                        "warning": (
                            "File could not be assigned "
                            "to a split."
                        )
                    }
                )

            split_plan["datasets"][dataset].append(
                {
                    "file": filename,
                    "group": group,
                    "split": split
                }
            )

            print(
                f"{group:20s} "
                f"{split:25s} "
                f"{filename}"
            )

            total_files += 1

    # -----------------------------------------------------
    # Important methodological notes
    # -----------------------------------------------------

    split_plan["methodology_notes"] = [

        (
            "The 70/15/15 values are target proportions, "
            "not guaranteed row-level proportions for "
            "group-based CIC splitting."
        ),

        (
            "CIC capture groups are kept intact to reduce "
            "the risk of closely related traffic appearing "
            "across train, validation and test."
        ),

        (
            "UNSW-NB15 official testing data is preserved "
            "as the final test source."
        ),

        (
            "UNSW-NB15 official training data will later "
            "be split into training and validation."
        ),

        (
            "Class coverage and actual row proportions "
            "must be validated before model training."
        ),

        (
            "Duplicate feature rows are not automatically "
            "removed. Their distribution across splits "
            "must be checked separately."
        )
    ]

    split_plan["summary"] = {
        "files_in_manifest": total_files,
        "warnings": len(
            split_plan["warnings"]
        )
    }

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
            split_plan,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 70)
    print("SPLIT PLAN CREATED")
    print("=" * 70)

    print(
        f"Files processed : {total_files}"
    )

    print(
        f"Warnings        : "
        f"{len(split_plan['warnings'])}"
    )

    print(
        f"Saved to        : {OUTPUT_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()