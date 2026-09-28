from pathlib import Path
import json
import pandas as pd


DATASETS = {
    "CIC-IDS2017": Path(
        "/Volumes/Ganesh's SD/Cybersecurity-Project/datasets/raw/cicids2017"
    ),
    "CIC-IDS2018": Path(
        "/Volumes/Ganesh's SD/Cybersecurity-Project/datasets/raw/cicids2018"
    ),
    "UNSW-NB15": Path(
        "/Volumes/Ganesh's SD/Cybersecurity-Project/datasets/raw/unsw_nb15"
    ),
}


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "dataset_schema.json"
)


def extract_schema():

    result = {
        "generated_by": "CyberSecAI Dataset Schema Extractor",
        "datasets": {}
    }

    for dataset_name, dataset_path in DATASETS.items():

        dataset_info = {
            "dataset_path": str(dataset_path),
            "file_count": 0,
            "files": []
        }

        csv_files = sorted(dataset_path.rglob("*.csv"))

        # Ignore macOS metadata files
        csv_files = [
            file for file in csv_files
            if not file.name.startswith("._")
        ]

        for csv_file in csv_files:

            try:

                # Read ONLY the header.
                # No dataset rows are loaded into memory.
                df = pd.read_csv(
                    csv_file,
                    nrows=0
                )

                columns = [
                    str(column).strip()
                    for column in df.columns
                ]

                file_info = {
                    "file_name": csv_file.name,
                    "file_path": str(csv_file),
                    "column_count": len(columns),
                    "columns": columns
                }

                dataset_info["files"].append(file_info)

            except Exception as error:

                dataset_info["files"].append({
                    "file_name": csv_file.name,
                    "file_path": str(csv_file),
                    "error": str(error)
                })

        dataset_info["file_count"] = len(dataset_info["files"])

        result["datasets"][dataset_name] = dataset_info

    return result


def main():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    schema = extract_schema()

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            schema,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\nDataset schema extraction completed.")
    print(f"Saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()