from pathlib import Path
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "CIC-IDS2017": PROJECT_ROOT / "data" / "processed" / "cicids2017",
    "CIC-IDS2018": PROJECT_ROOT / "data" / "processed" / "cicids2018",
    "UNSW-NB15": PROJECT_ROOT / "data" / "processed" / "unswnb15",
}

OUTPUT = PROJECT_ROOT / "data" / "metadata" / "unified_schema.json"


def discover_csv_files(directory):
    if not directory.exists():
        return []

    return sorted(
        file for file in directory.glob("*.csv")
        if not file.name.startswith("._")
    )


def inspect_file(path):
    print(f"  Reading: {path.name}")

    df = pd.read_csv(path, nrows=1000, low_memory=False)

    return {
        "file": path.name,
        "rows_sampled": len(df),
        "columns": list(df.columns),
        "column_count": len(df.columns),
        "dtypes": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },
        "numeric_columns": list(
            df.select_dtypes(include="number").columns
        ),
        "categorical_columns": list(
            df.select_dtypes(exclude="number").columns
        ),
    }


def main():

    schema = {
        "project": "CyberSecAI",
        "datasets": {},
        "common_columns": [],
        "dataset_specific_columns": {},
    }

    column_sets = {}

    print("\n===================================")
    print(" CyberSecAI Unified Schema Analysis")
    print("===================================\n")

    for dataset, directory in DATASETS.items():

        files = discover_csv_files(directory)

        print(f"{dataset}: {len(files)} CSV files")

        schema["datasets"][dataset] = []

        dataset_columns = set()

        for file in files:

            info = inspect_file(file)

            schema["datasets"][dataset].append(info)

            dataset_columns.update(info["columns"])

        column_sets[dataset] = dataset_columns

    # Only compare datasets that actually contain columns
    valid_sets = [
        columns
        for columns in column_sets.values()
        if columns
    ]

    if len(valid_sets) >= 2:

        common_columns = set.intersection(*valid_sets)

        schema["common_columns"] = sorted(common_columns)

        for dataset, columns in column_sets.items():

            schema["dataset_specific_columns"][dataset] = sorted(
                columns - common_columns
            )

    elif len(valid_sets) == 1:

        schema["common_columns"] = sorted(valid_sets[0])

        for dataset, columns in column_sets.items():
            schema["dataset_specific_columns"][dataset] = []

    # Save result
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    print("\n===================================")
    print(" Analysis Complete")
    print("===================================\n")

    for dataset, files in schema["datasets"].items():

        if files:
            print(
                f"{dataset}: "
                f"{len(files)} files | "
                f"{len(column_sets[dataset])} unique columns"
            )
        else:
            print(f"{dataset}: 0 files")

    print(
        f"\nCommon columns across datasets: "
        f"{len(schema['common_columns'])}"
    )

    for dataset, columns in schema[
        "dataset_specific_columns"
    ].items():

        print(
            f"{dataset} dataset-specific columns: "
            f"{len(columns)}"
        )

    print(f"\nSaved to:")
    print(OUTPUT)


if __name__ == "__main__":
    main()