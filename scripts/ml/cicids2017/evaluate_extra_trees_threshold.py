from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = ROOT / "models" / "cicids2017" / "extra_trees_binary.joblib"

VALIDATION_ROOT = (
    ROOT / "data" / "splits" / "cicids2017" / "validation"
)

RESULTS_DIR = (
    ROOT / "results" / "ml" / "cicids2017"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = (
    RESULTS_DIR / "extra_trees_threshold_005_evaluation.json"
)

THRESHOLD = 0.05

EXCLUDED_COLUMNS = {
    "Label",
    "canonical_label",
    "binary_label",
    "original_label",
    "attack_cat",
}


def get_feature_columns():

    files = sorted(VALIDATION_ROOT.glob("*.csv"))

    df = pd.read_csv(
        files[0],
        nrows=5
    )

    return [
        column
        for column in df.columns
        if (
            column not in EXCLUDED_COLUMNS
            and pd.api.types.is_numeric_dtype(df[column])
        )
    ]


def load_validation(feature_columns):

    X_parts = []
    y_parts = []

    files = sorted(VALIDATION_ROOT.glob("*.csv"))

    print("Loading validation data...")

    for i, file in enumerate(files, 1):

        print(
            f"{i:02d}/{len(files)} {file.name}"
        )

        df = pd.read_csv(
            file,
            usecols=feature_columns + ["binary_label"]
        )

        X = df[feature_columns].apply(
            pd.to_numeric,
            errors="coerce"
        )

        y = pd.to_numeric(
            df["binary_label"],
            errors="coerce"
        )

        X = X.replace(
            [np.inf, -np.inf],
            np.nan
        )

        valid = (
            X.notna().all(axis=1)
            & y.notna()
        )

        X_parts.append(
            X.loc[valid].astype(np.float32)
        )

        y_parts.append(
            y.loc[valid].astype(np.int8)
        )

    return (
        pd.concat(X_parts, ignore_index=True),
        pd.concat(y_parts, ignore_index=True),
    )


def main():

    print("=" * 70)
    print("EXTRA TREES THRESHOLD 0.05 EVALUATION")
    print("=" * 70)

    feature_columns = get_feature_columns()

    print(f"Features: {len(feature_columns)}")

    print("\nLoading model...")

    model = joblib.load(
        MODEL_PATH
    )

    X_val, y_val = load_validation(
        feature_columns
    )

    print(
        f"\nValidation rows: {len(X_val):,}"
    )

    print("\nGenerating probabilities...")

    probabilities = model.predict_proba(
        X_val
    )[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(np.int8)

    accuracy = accuracy_score(
        y_val,
        predictions
    )

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

    roc_auc = roc_auc_score(
        y_val,
        probabilities
    )

    cm = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn)

    fnr = fn / (fn + tp)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"Threshold           : {THRESHOLD}")
    print(f"Accuracy            : {accuracy:.6f}")
    print(f"Precision           : {precision:.6f}")
    print(f"Recall              : {recall:.6f}")
    print(f"F1 Score            : {f1:.6f}")
    print(f"ROC-AUC             : {roc_auc:.6f}")

    print("\nCONFUSION MATRIX")
    print("-" * 50)

    print("                 Predicted")
    print("                 Benign     Attack")
    print(f"Actual Benign    {tn:8,d} {fp:10,d}")
    print(f"Actual Attack    {fn:8,d} {tp:10,d}")

    print("\nSecurity metrics:")
    print(f"False Positive Rate: {fpr:.6f}")
    print(f"False Negative Rate: {fnr:.6f}")

    print("\nClassification Report")
    print("-" * 70)

    report_text = classification_report(
        y_val,
        predictions,
        target_names=["BENIGN", "ATTACK"],
        zero_division=0
    )

    print(report_text)

    report_dict = classification_report(
        y_val,
        predictions,
        target_names=["BENIGN", "ATTACK"],
        output_dict=True,
        zero_division=0
    )

    results = {
        "dataset": "CIC-IDS2017",
        "model": "ExtraTrees",
        "task": "binary_detection",
        "threshold": THRESHOLD,

        "validation_rows": int(len(y_val)),
        "feature_count": int(len(feature_columns)),

        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "false_positive_rate": float(fpr),
            "false_negative_rate": float(fnr),
        },

        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },

        "classification_report": report_dict,
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print("\nJSON saved:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("EXTRA TREES THRESHOLD EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()