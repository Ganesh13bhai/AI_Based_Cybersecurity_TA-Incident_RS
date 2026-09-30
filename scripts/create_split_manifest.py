import json
import glob
import os
import re


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PROCESSED_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "metadata",
    "split_manifest.json"
)


def detect_dataset(path):
    path_lower = path.lower()

    if "cicids2017" in path_lower:
        return "CIC-IDS2017"

    if "cicids2018" in path_lower:
        return "CIC-IDS2018"

    if "unswnb15" in path_lower:
        return "UNSW-NB15"

    return "UNKNOWN"


def detect_group(dataset, filename):
    name = filename.lower()

    if dataset == "CIC-IDS2017":

        if name.startswith("monday"):
            return "Monday"

        if name.startswith("tuesday"):
            return "Tuesday"

        if name.startswith("wednesday"):
            return "Wednesday"

        if name.startswith("thursday"):
            return "Thursday"

        if name.startswith("friday"):
            return "Friday"

        return "UNKNOWN"

    if dataset == "CIC-IDS2018":

        if name.startswith("benign-monday"):
            return "Monday"

        if name.startswith("bruteforce-tuesday"):
            return "Tuesday"

        if name.startswith("dos-wednesday"):
            return "Wednesday"

        if name.startswith("infiltration-thursday"):
            return "Thursday"

        if name.startswith("webattacks-thursday"):
            return "Thursday"

        if name.startswith("ddos-friday"):
            return "Friday"

        if name.startswith("portscan-friday"):
            return "Friday"

        if name.startswith("botnet-friday"):
            return "Friday"

        return "UNKNOWN"

    if dataset == "UNSW-NB15":

        if "training" in name:
            return "official_train"

        if "testing" in name:
            return "official_test"

        return "UNKNOWN"

    return "UNKNOWN"


def main():

    print("=" * 70)
    print("CyberSecAI - Split Manifest Generator")
    print("=" * 70)

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

    files = [
        f
        for f in files
        if not os.path.basename(f).startswith("._")
    ]

    manifest = {
        "version": "1.0",
        "description": (
            "Dataset grouping manifest for "
            "leakage-aware train/validation/test splitting."
        ),
        "datasets": {}
    }

    for file in files:

        dataset = detect_dataset(file)

        if dataset == "UNKNOWN":
            continue

        filename = os.path.basename(file)

        group = detect_group(
            dataset,
            filename
        )

        if dataset not in manifest["datasets"]:
            manifest["datasets"][dataset] = []

        manifest["datasets"][dataset].append(
            {
                "file": filename,
                "group": group
            }
        )

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
            manifest,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\nDataset groups:")

    for dataset, items in manifest["datasets"].items():

        print(f"\n{dataset}")

        for item in items:
            print(
                f"  {item['group']:18s} "
                f"{item['file']}"
            )

    print("\n" + "=" * 70)
    print("Split manifest created")
    print("=" * 70)

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()