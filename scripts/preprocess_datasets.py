from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

RAW = Path("/Volumes/Ganesh's SD/Cybersecurity-Project/datasets/raw")
OUT = ROOT / "data" / "processed"
REPORT = ROOT / "data" / "metadata" / "preprocessing_report.json"

DATASETS = {
    "CIC-IDS2017": RAW / "cicids2017",
    "CIC-IDS2018": RAW / "cicids2018",
    "UNSW-NB15": RAW / "unsw_nb15",
}

LABEL_MAP = {
    "benign": "BENIGN",
    "normal": "BENIGN",
    "ddos": "DDOS",
    "dos": "DOS",
    "dos hulk": "DOS",
    "dos goldeneye": "DOS",
    "dos slowloris": "DOS",
    "dos slowhttptest": "DOS",
    "portscan": "PORT_SCAN",
    "portscan": "PORT_SCAN",
    "bot": "BOTNET",
    "botnet": "BOTNET",
    "infiltration": "INFILTRATION",
    "ftp-patator": "BRUTE_FORCE",
    "ssh-patator": "BRUTE_FORCE",
    "bruteforce": "BRUTE_FORCE",
    "web attack - brute force": "WEB_ATTACK",
    "web attack - xss": "WEB_ATTACK",
    "web attack - sql injection": "WEB_ATTACK",
    "webattacks": "WEB_ATTACK",
    "heartbleed": "OTHER_ATTACK",
    "generic": "OTHER_ATTACK",
    "exploits": "EXPLOIT",
    "fuzzers": "FUZZER",
    "reconnaissance": "RECONNAISSANCE",
    "shellcode": "SHELLCODE",
    "analysis": "ANALYSIS",
    "backdoor": "BACKDOOR",
    "worms": "WORM",
}

def clean_text(value):
    return " ".join(
        str(value).strip().lower()
        .replace("�", "-")
        .replace("–", "-")
        .replace("—", "-")
        .split()
    )

def find_label_column(columns):
    normalized = {str(c).strip().lower(): c for c in columns}

    for candidate in ["attack_cat", "label"]:
        if candidate in normalized:
            return normalized[candidate]

    return None

def normalize_label(value):
    key = clean_text(value)
    return LABEL_MAP.get(key, "OTHER_ATTACK")

def process_file(dataset, source_file, output_file):
    df = pd.read_csv(source_file, low_memory=False)

    original_rows = len(df)

    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    label_column = find_label_column(df.columns)

    if label_column is None:
        raise ValueError(f"No label column found in {source_file}")

    # Normalize labels
    df["canonical_label"] = df[label_column].apply(normalize_label)

    # Binary label
    df["binary_label"] = (df["canonical_label"] != "BENIGN").astype(int)

    # Convert numeric-looking columns
    numeric_columns = []

    for column in df.columns:
        if column in [label_column, "canonical_label"]:
            continue

        converted = pd.to_numeric(df[column], errors="coerce")

        if converted.notna().sum() > 0:
            df[column] = converted
            numeric_columns.append(column)

    # Replace infinity with NaN
    infinity_count = 0

    for column in numeric_columns:
        mask = np.isinf(df[column])
        infinity_count += int(mask.sum())
        df.loc[mask, column] = np.nan

    # Missing values before filling
    missing_before = int(df[numeric_columns].isna().sum().sum())

    # Median imputation for numeric features
    for column in numeric_columns:
        if df[column].isna().any():
            median = df[column].median()

            if pd.notna(median):
                df[column] = df[column].fillna(median)
            else:
                df[column] = df[column].fillna(0)

    missing_after = int(df[numeric_columns].isna().sum().sum())

    # Remove exact duplicates within this source file
    before_dedup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = before_dedup - len(df)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    return {
        "dataset": dataset,
        "source_file": str(source_file),
        "output_file": str(output_file),
        "original_rows": original_rows,
        "final_rows": len(df),
        "duplicates_removed": duplicates_removed,
        "infinity_values_replaced": infinity_count,
        "missing_values_before_imputation": missing_before,
        "missing_values_after_imputation": missing_after,
        "numeric_feature_count": len(numeric_columns),
        "label_column": label_column,
        "label_distribution": df["canonical_label"].value_counts().to_dict(),
    }

def main():
    print("=" * 80)
    print("CYBERSECAI DATA PREPROCESSING")
    print("=" * 80)

    OUT.mkdir(parents=True, exist_ok=True)

    report = {
        "project": "CyberSecAI",
        "raw_data_modified": False,
        "datasets": {},
    }

    for dataset, dataset_path in DATASETS.items():

        print(f"\n{'=' * 80}")
        print(f"DATASET: {dataset}")
        print(f"{'=' * 80}")

        dataset_output = OUT / dataset.lower().replace("-", "").replace(" ", "_")
        dataset_output.mkdir(parents=True, exist_ok=True)

        report["datasets"][dataset] = []

        files = sorted(
            f for f in dataset_path.glob("*.csv")
            if not f.name.startswith("._")
        )

        for source_file in files:

            # Preserve UNSW training/testing as separate files
            output_file = dataset_output / source_file.name

            print(f"\nProcessing: {source_file.name}")

            result = process_file(
                dataset,
                source_file,
                output_file
            )

            report["datasets"][dataset].append(result)

            print(f"  Original rows : {result['original_rows']:,}")
            print(f"  Final rows    : {result['final_rows']:,}")
            print(f"  Duplicates    : {result['duplicates_removed']:,}")
            print(f"  Inf replaced  : {result['infinity_values_replaced']:,}")
            print(f"  Missing before: {result['missing_values_before_imputation']:,}")
            print(f"  Missing after : {result['missing_values_after_imputation']:,}")

    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"Report: {REPORT}")
    print(f"Output: {OUT}")
    print("Raw datasets were NOT modified.")

if __name__ == "__main__":
    main()
