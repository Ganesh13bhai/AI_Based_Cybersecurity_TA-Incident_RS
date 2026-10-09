import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    ROOT
    / "models"
    / "cicids2017"
    / "extra_trees_binary.joblib"
)

TEST_ROOT = (
    ROOT
    / "data"
    / "splits"
    / "cicids2017"
    / "test"
)

RESULT_ROOT = (
    ROOT
    / "results"
    / "ml"
    / "cicids2017"
)

RESULT_ROOT.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.05

TARGET_COLUMNS = {
    "Label",
    "canonical_label",
    "binary_label",
    "original_label",
    "attack_cat",
}


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    files = sorted(TEST_ROOT.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No test CSV files found in {TEST_ROOT}"
        )

    X_parts = []
    y_parts = []

    print("\nLoading untouched CIC-IDS2017 test data...")
    print("=" * 70)

    for i, file in enumerate(files, 1):

        print(
            f"{i:02d}/{len(files)} "
            f"{file.name}"
        )

        df = pd.read_csv(file)

        feature_columns = [
            c for c in df.columns
            if c not in TARGET_COLUMNS
        ]

        X = df[feature_columns].apply(
            pd.to_numeric,
            errors="coerce"
        )

        X = X.replace(
            [np.inf, -np.inf],
            np.nan
        )

        y = pd.to_numeric(
            df["binary_label"],
            errors="coerce"
        )

        valid = (
            X.notna().all(axis=1)
            & y.notna()
        )

        X = X.loc[valid]
        y = y.loc[valid]

        X_parts.append(X)
        y_parts.append(y)

    X_test = pd.concat(
        X_parts,
        ignore_index=True
    )

    y_test = pd.concat(
        y_parts,
        ignore_index=True
    )

    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.int8)

    return X_test, y_test


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINAL CIC-IDS2017 EXTRA TREES TEST EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not TEST_ROOT.exists():

        raise FileNotFoundError(
            f"Test directory not found:\n{TEST_ROOT}"
        )

    print("\nModel:")
    print(MODEL_PATH)

    print("\nThreshold:")
    print(THRESHOLD)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading Extra Trees model...")

    model = joblib.load(MODEL_PATH)

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    X_test, y_test = load_test_data()

    print("=" * 70)

    print(
        f"Test rows : {len(X_test):,}"
    )

    print(
        f"Features  : {X_test.shape[1]}"
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print("\nGenerating test predictions...")

    start_time = time.time()

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    prediction_time = time.time() - start_time

    y_pred = (
        probabilities >= THRESHOLD
    ).astype(np.int8)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "BENIGN",
            "ATTACK"
        ],
        output_dict=True,
        zero_division=0
    )

    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL CIC-IDS2017 TEST RESULTS")
    print("=" * 70)

    print(
        f"Threshold            : {THRESHOLD:.2f}"
    )

    print(
        f"Accuracy             : {accuracy:.6f}"
    )

    print(
        f"Precision            : {precision:.6f}"
    )

    print(
        f"Recall               : {recall:.6f}"
    )

    print(
        f"F1 Score             : {f1:.6f}"
    )

    print(
        f"ROC-AUC              : {roc_auc:.6f}"
    )

    print("\nConfusion Matrix")
    print("-" * 50)

    print(
        f"True Negatives       : {tn:,}"
    )

    print(
        f"False Positives      : {fp:,}"
    )

    print(
        f"False Negatives      : {fn:,}"
    )

    print(
        f"True Positives       : {tp:,}"
    )

    print(
        f"\nFalse Positive Rate  : {fpr:.6f}"
    )

    print(
        f"False Negative Rate  : {fnr:.6f}"
    )

    print(
        f"Prediction Time      : {prediction_time:.4f} sec"
    )

    print("\nClassification Report")
    print("-" * 70)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "BENIGN",
                "ATTACK"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    result = {
        "dataset": "CIC-IDS2017",
        "model": "Extra Trees",
        "task": "binary_classification",
        "threshold": THRESHOLD,

        "test_rows": int(len(y_test)),
        "features": int(X_test.shape[1]),

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

        "prediction_time_seconds": float(
            prediction_time
        ),

        "classification_report": report,

        "model_path": str(
            MODEL_PATH
        ),

        "test_path": str(
            TEST_ROOT
        ),
    }

    json_path = (
        RESULT_ROOT
        / "final_extra_trees_test_evaluation.json"
    )

    with open(
        json_path,
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Confusion matrix image
    # --------------------------------------------------------

    cm = np.array([
        [tn, fp],
        [fn, tp]
    ])

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "BENIGN",
            "ATTACK"
        ]
    )

    display.plot(
        ax=ax,
        values_format=","
    )

    ax.set_title(
        "CIC-IDS2017 — Extra Trees\n"
        f"Final Test Confusion Matrix (Threshold={THRESHOLD})"
    )

    plt.tight_layout()

    image_path = (
        RESULT_ROOT
        / "final_extra_trees_test_confusion_matrix.png"
    )

    plt.savefig(
        image_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL EVALUATION COMPLETE")
    print("=" * 70)

    print("\nJSON saved:")
    print(json_path)

    print("\nConfusion matrix saved:")
    print(image_path)

    print("\nFINAL MODEL:")
    print("Extra Trees")

    print("\nFINAL THRESHOLD:")
    print("0.05")

    print("\nIMPORTANT:")
    print(
        "These results are from the untouched CIC-IDS2017 test set."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
