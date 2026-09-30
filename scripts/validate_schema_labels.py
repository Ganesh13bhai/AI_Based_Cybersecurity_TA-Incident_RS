import json
from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = Path(
    "/Volumes/Ganesh's SD/Cybersecurity-Project/datasets/raw"
)

MAPPING_FILE = (
    ROOT / "data" / "metadata" / "schema_label_mapping.json"
)

OUTPUT_FILE = (
    ROOT / "data" / "metadata" / "schema_label_validation.json"
)


# ============================================================
# DATASET LOCATIONS
# ============================================================

DATASET_PATHS = {
    "CIC-IDS2017": DATASET_ROOT / "cicids2017",
    "CIC-IDS2018": DATASET_ROOT / "cicids2018",
    "UNSW-NB15": DATASET_ROOT / "unsw_nb15",
}


# ============================================================
# LOAD MAPPING
# ============================================================

with open(MAPPING_FILE, "r", encoding="utf-8") as f:
    mapping = json.load(f)


# ============================================================
# HELPERS
# ============================================================

def clean_label(value):
    """
    Normalize labels so minor encoding differences
    do not create separate attack classes.
    """

    label = str(value).strip().lower()

    # Normalize corrupted Web Attack separator
    label = label.replace("�", "-")
    label = label.replace("–", "-")
    label = label.replace("—", "-")

    # Normalize repeated spaces
    label = " ".join(label.split())

    return label


def find_label_column(columns, dataset_name):
    """
    Find the correct label column.
    """

    normalized = {
        str(col).strip().lower(): col
        for col in columns
    }

    if dataset_name == "UNSW-NB15":

        # UNSW has attack_cat and label.
        if "attack_cat" in normalized:
            return normalized["attack_cat"]

        if "label" in normalized:
            return normalized["label"]

    else:

        if "label" in normalized:
            return normalized["label"]

    return None


# ============================================================
# VALIDATE ONE DATASET
# ============================================================

def validate_dataset(dataset_name, config):

    print("\n" + "=" * 70)
    print(dataset_name)
    print("=" * 70)

    dataset_path = DATASET_PATHS[dataset_name]

    print(f"Dataset path:")
    print(dataset_path)

    if not dataset_path.exists():

        print("❌ Dataset directory does not exist")

        return {
            "status": "DIRECTORY_NOT_FOUND",
            "path": str(dataset_path),
            "files": []
        }

    csv_files = sorted(
        [
            file
            for file in dataset_path.iterdir()
            if file.is_file()
            and file.suffix.lower() == ".csv"
            and not file.name.startswith("._")
        ]
    )

    if not csv_files:

        print("❌ No CSV files found")

        return {
            "status": "NO_CSV_FILES",
            "path": str(dataset_path),
            "files": []
        }

    print(f"CSV files found: {len(csv_files)}")

    # --------------------------------------------------------
    # Mapping
    # --------------------------------------------------------

    label_mapping = config["label_mapping"]

    normalized_mapping = {
        clean_label(key): value
        for key, value in label_mapping.items()
    }

    file_results = []

    # --------------------------------------------------------
    # Process every CSV
    # --------------------------------------------------------

    for file in csv_files:

        print("\n" + "-" * 70)
        print(f"FILE: {file.name}")
        print("-" * 70)

        try:

            # Read header only first
            header = pd.read_csv(
                file,
                nrows=0
            )

            columns = list(header.columns)

            print(f"Columns: {len(columns)}")

            label_column = find_label_column(
                columns,
                dataset_name
            )

            if label_column is None:

                print("❌ Label column NOT found")

                file_results.append({
                    "file": file.name,
                    "status": "LABEL_COLUMN_NOT_FOUND",
                    "column_count": len(columns),
                    "columns": columns
                })

                continue

            print(f"Label column: {label_column}")

            # ------------------------------------------------
            # Read labels only
            # ------------------------------------------------

            labels = pd.read_csv(
                file,
                usecols=[label_column]
            )[label_column]

            labels = (
                labels
                .dropna()
                .astype(str)
                .str.strip()
            )

            unique_labels = sorted(
                labels.unique().tolist()
            )

            print(f"Unique labels: {len(unique_labels)}")

            # ------------------------------------------------
            # Check labels against mapping
            # ------------------------------------------------

            unknown_labels = []

            for label in unique_labels:

                if clean_label(label) not in normalized_mapping:

                    unknown_labels.append(label)

            if unknown_labels:

                print("\n❌ UNKNOWN LABELS:")

                for label in unknown_labels:
                    print(f"   → {label}")

                status = "UNKNOWN_LABELS"

            else:

                print("\n✅ All labels recognized")

                status = "VALID"

            # ------------------------------------------------
            # Print label information
            # ------------------------------------------------

            print("\nLabels found:")

            for label in unique_labels:

                mapped = normalized_mapping.get(
                    clean_label(label)
                )

                if mapped:

                    print(
                        f"   {label}"
                        f" → binary={mapped['binary_label']}"
                        f", family={mapped['attack_family']}"
                    )

                else:

                    print(
                        f"   {label}"
                        f" → UNKNOWN"
                    )

            # ------------------------------------------------
            # Save result
            # ------------------------------------------------

            file_results.append({

                "file": file.name,

                "status": status,

                "label_column": label_column,

                "column_count": len(columns),

                "unique_labels": unique_labels,

                "unknown_labels": unknown_labels

            })

        except Exception as e:

            print(f"\n❌ ERROR: {e}")

            file_results.append({

                "file": file.name,

                "status": "ERROR",

                "error": str(e)

            })

    # --------------------------------------------------------
    # Dataset status
    # --------------------------------------------------------

    dataset_status = (

        "VALID"

        if file_results
        and all(
            result["status"] == "VALID"
            for result in file_results
        )

        else "CHECK_REQUIRED"
    )

    return {

        "status": dataset_status,

        "path": str(dataset_path),

        "file_count": len(file_results),

        "files": file_results

    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 70)
    print("CyberSecAI")
    print("SCHEMA / LABEL VALIDATION")
    print("=" * 70)

    final_result = {

        "project": "CyberSecAI",

        "mapper_version": mapping.get(
            "version",
            "unknown"
        ),

        "dataset_root": str(DATASET_ROOT),

        "datasets": {}

    }

    # --------------------------------------------------------
    # Validate all datasets
    # --------------------------------------------------------

    for dataset_name, config in mapping["datasets"].items():

        result = validate_dataset(
            dataset_name,
            config
        )

        final_result["datasets"][dataset_name] = result

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_result,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n\n")

    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    all_valid = True

    for dataset_name, result in final_result["datasets"].items():

        if result["status"] == "VALID":

            print(
                f"✅ {dataset_name}: VALID"
            )

        else:

            print(
                f"⚠️ {dataset_name}: CHECK REQUIRED"
            )

            all_valid = False

    print("\n")

    if all_valid:

        print(
            "🎉 SCHEMA / LABEL MAPPING VALIDATION PASSED"
        )

    else:

        print(
            "⚠️ Schema/label issues require correction."
        )

    print("\nValidation report:")

    print(OUTPUT_FILE)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()