import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = ROOT / "models" / "cicids2017" / "logistic_regression_binary.joblib"
SPLIT_ROOT = ROOT / "data" / "splits" / "cicids2017" / "validation"

THRESHOLDS = np.arange(0.05, 0.91, 0.05)

TARGET_COLUMNS = {
    "Label",
    "canonical_label",
    "binary_label",
    "original_label",
    "attack_cat",
}


def get_validation_files():
    return sorted(SPLIT_ROOT.glob("*.csv"))


def load_validation_data():
    files = get_validation_files()

    if not files:
        raise FileNotFoundError(
            f"No validation CSV files found in {SPLIT_ROOT}"
        )

    X_parts = []
    y_parts = []

    for i, file in enumerate(files, 1):
        print(f"{i:02d}/{len(files)} {file.name}")

        df = pd.read_csv(file)

        feature_columns = [
            c for c in df.columns
            if c not in TARGET_COLUMNS
        ]

        X = df[feature_columns].apply(
            pd.to_numeric,
            errors="coerce"
        )

        X = X.replace([np.inf, -np.inf], np.nan)

        y = pd.to_numeric(
            df["binary_label"],
            errors="coerce"
        )

        valid = X.notna().all(axis=1) & y.notna()

        X_parts.append(X.loc[valid])
        y_parts.append(y.loc[valid])

    X = pd.concat(X_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True)

    return X.astype(np.float32), y.astype(np.int8)


def main():
    print("=" * 70)
    print("CIC-IDS2017 LOGISTIC REGRESSION THRESHOLD ANALYSIS")
    print("=" * 70)

    if not MODEL_PATH.exists():
        print(f"\nERROR: Model not found:")
        print(MODEL_PATH)
        sys.exit(1)

    print("\nLoading Logistic Regression model...")
    model = joblib.load(MODEL_PATH)

    print("Loading validation data...")
    X_val, y_val = load_validation_data()

    print("=" * 70)
    print(f"Validation rows: {len(X_val):,}")
    print(f"Features: {X_val.shape[1]}")

    print("\nGenerating attack probabilities...")

    # Pipeline returns probability for class 1 = ATTACK
    probabilities = model.predict_proba(X_val)[:, 1]

    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)

    print(
        f"{'Threshold':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
    )

    print("-" * 48)

    results = []

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_val,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_val,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_val,
            predictions,
            zero_division=0
        )

        results.append({
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        })

        print(
            f"{threshold:<12.2f}"
            f"{precision:<12.4f}"
            f"{recall:<12.4f}"
            f"{f1:<12.4f}"
        )

    best = max(results, key=lambda x: x["f1"])

    print("\n" + "=" * 70)
    print("BEST LOGISTIC REGRESSION THRESHOLD")
    print("=" * 70)

    print(f"Threshold : {best['threshold']:.2f}")
    print(f"Precision : {best['precision']:.4f}")
    print(f"Recall    : {best['recall']:.4f}")
    print(f"F1 Score  : {best['f1']:.4f}")

    output_dir = ROOT / "results" / "ml" / "cicids2017"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir /
        "logistic_regression_threshold_analysis.json"
    )

    output = {
        "model": "Logistic Regression",
        "dataset": "CIC-IDS2017",
        "validation_rows": int(len(X_val)),
        "features": int(X_val.shape[1]),
        "thresholds": results,
        "best_threshold": best
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\nResults saved:")
    print(output_path)

    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
