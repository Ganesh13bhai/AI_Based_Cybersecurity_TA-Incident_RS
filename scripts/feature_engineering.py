from pathlib import Path
import pandas as pd
import numpy as np
import json

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed"
OUTPUT = ROOT / "data" / "features"
REPORT = ROOT / "data" / "metadata" / "feature_engineering_report.json"

DATASETS = ["cicids2017", "cicids2018", "unsw_nb15"]

def safe_divide(a, b):
    return a / b.replace(0, np.nan)

def engineer_cic(df):
    # Packet ratios
    if "Total Fwd Packets" in df and "Total Backward Packets" in df:
        df["Fwd_Bwd_Packet_Ratio"] = safe_divide(
            df["Total Fwd Packets"],
            df["Total Backward Packets"]
        )

    # Byte ratios
    if "Total Length of Fwd Packets" in df and "Total Length of Bwd Packets" in df:
        df["Fwd_Bwd_Byte_Ratio"] = safe_divide(
            df["Total Length of Fwd Packets"],
            df["Total Length of Bwd Packets"]
        )

    # Packet-rate ratio
    if "Fwd Packets/s" in df and "Bwd Packets/s" in df:
        df["Fwd_Bwd_PacketRate_Ratio"] = safe_divide(
            df["Fwd Packets/s"],
            df["Bwd Packets/s"]
        )

    # Average bytes per packet
    if "Total Length of Fwd Packets" in df and "Total Fwd Packets" in df:
        df["Avg_Fwd_Bytes_Per_Packet"] = safe_divide(
            df["Total Length of Fwd Packets"],
            df["Total Fwd Packets"]
        )

    if "Total Length of Bwd Packets" in df and "Total Backward Packets" in df:
        df["Avg_Bwd_Bytes_Per_Packet"] = safe_divide(
            df["Total Length of Bwd Packets"],
            df["Total Backward Packets"]
        )

    # Flow packet/byte rates
    if "Flow Duration" in df:
        duration_seconds = df["Flow Duration"].replace(0, np.nan) / 1_000_000

        if "Total Fwd Packets" in df and "Total Backward Packets" in df:
            df["Calculated_Packets_Per_Second"] = (
                df["Total Fwd Packets"] + df["Total Backward Packets"]
            ) / duration_seconds

        if "Total Length of Fwd Packets" in df and "Total Length of Bwd Packets" in df:
            df["Calculated_Bytes_Per_Second"] = (
                df["Total Length of Fwd Packets"] +
                df["Total Length of Bwd Packets"]
            ) / duration_seconds

    # TCP flag aggregate
    flag_columns = [
        "FIN Flag Count",
        "SYN Flag Count",
        "RST Flag Count",
        "PSH Flag Count",
        "ACK Flag Count",
        "URG Flag Count",
        "CWE Flag Count",
        "ECE Flag Count",
    ]

    existing_flags = [c for c in flag_columns if c in df.columns]

    if existing_flags:
        df["Total_TCP_Flag_Count"] = df[existing_flags].sum(axis=1)

    # Replace generated infinity values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df

def process_dataset(dataset):
    input_dir = INPUT / dataset
    output_dir = OUTPUT / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for file in sorted(input_dir.glob("*.csv")):
        if file.name.startswith("._"):
            continue
        print(f"Processing: {file.name}")

        df = pd.read_csv(file, low_memory=False)

        original_columns = len(df.columns)
        original_rows = len(df)

        if dataset.startswith("cicids"):
            df = engineer_cic(df)

        # Remove rows created with invalid derived values
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        inf_count = np.isinf(
            df[numeric_columns].to_numpy()
        ).sum()

        nan_before = int(df[numeric_columns].isna().sum().sum())

        # Median-fill only derived NaN values / numeric missing values
        for column in numeric_columns:
            if df[column].isna().any():
                median = df[column].median()

                if pd.notna(median):
                    df[column] = df[column].fillna(median)

        nan_after = int(df[numeric_columns].isna().sum().sum())

        output_file = output_dir / file.name
        df.to_csv(output_file, index=False)

        results.append({
            "file": file.name,
            "original_rows": original_rows,
            "final_rows": len(df),
            "original_columns": original_columns,
            "final_columns": len(df.columns),
            "new_features": len(df.columns) - original_columns,
            "infinity_values": int(inf_count),
            "numeric_missing_before": nan_before,
            "numeric_missing_after": nan_after,
        })

        print(f"  Rows    : {len(df):,}")
        print(f"  Columns : {len(df.columns)}")
        print(f"  New     : {len(df.columns) - original_columns}")

    return results

def main():
    print("=" * 80)
    print("CYBERSECAI FEATURE ENGINEERING")
    print("=" * 80)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    report = {
        "project": "CyberSecAI",
        "datasets": {}
    }

    for dataset in DATASETS:
        print(f"\n{'=' * 80}")
        print(f"DATASET: {dataset}")
        print(f"{'=' * 80}")

        report["datasets"][dataset] = process_dataset(dataset)

    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 80)
    print(f"Output : {OUTPUT}")
    print(f"Report : {REPORT}")

if __name__ == "__main__":
    main()
