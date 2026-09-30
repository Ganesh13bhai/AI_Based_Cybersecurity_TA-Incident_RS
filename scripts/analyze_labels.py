import json
from pathlib import Path
import pandas as pd


DATASETS = {
    "CIC-IDS2017": Path("data/processed/cicids2017"),
    "CIC-IDS2018": Path("data/processed/cicids2018"),
    "UNSW-NB15": Path("data/processed/unswnb15"),
}

LABEL_COLUMNS = [
    "Label",
    "attack_cat",
    "label",
    "canonical_label",
    "binary_label",
]


inventory = {
    "version": "1.0",
    "description": "Observed label inventory from processed CyberSecAI datasets",
    "datasets": {}
}


for dataset, folder in DATASETS.items():

    print()
    print("=" * 80)
    print(dataset)
    print("=" * 80)

    dataset_info = {
        "directory": str(folder),
        "files": [],
        "all_labels": {},
    }

    files = sorted(
        f for f in folder.glob("*.csv")
        if not f.name.startswith("._")
    )

    for file in files:

        print(f"\nFile: {file.name}")

        # Only inspect label columns
        df = pd.read_csv(
            file,
            usecols=lambda c: c in LABEL_COLUMNS
        )

        file_info = {
            "file": file.name,
            "label_columns": {}
        }

        for column in df.columns:

            values = (
                df[column]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
            )

            values = sorted(values)

            file_info["label_columns"][column] = {
                "unique_count": len(values),
                "values": values
            }

            print(f"  {column}: {len(values)} unique values")
            print(f"    {values}")

            # Store global inventory
            if column not in dataset_info["all_labels"]:
                dataset_info["all_labels"][column] = set()

            dataset_info["all_labels"][column].update(values)

        dataset_info["files"].append(file_info)

    # Convert sets to sorted lists for JSON
    for column in dataset_info["all_labels"]:
        dataset_info["all_labels"][column] = sorted(
            dataset_info["all_labels"][column]
        )

    inventory["datasets"][dataset] = dataset_info


# Save JSON
output_path = Path("data/metadata/label_inventory.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(inventory, f, indent=2, ensure_ascii=False)


print()
print("=" * 80)
print("LABEL INVENTORY COMPLETE")
print("=" * 80)

for dataset, info in inventory["datasets"].items():

    print(f"\n{dataset}")

    for column, labels in info["all_labels"].items():
        print(
            f"  {column}: "
            f"{len(labels)} unique values"
        )

print()
print(f"Saved to: {output_path}")