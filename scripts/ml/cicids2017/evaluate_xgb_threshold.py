from pathlib import Path
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    ROOT
    / "models"
    / "cicids2017"
    / "xgboost_binary.json"
)

VALIDATION_ROOT = (
    ROOT
    / "data"
    / "splits"
    / "cicids2017"
    / "validation"
)

RESULTS_DIR = (
    ROOT
    / "results"
    / "ml"
    / "cicids2017"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

JSON_OUTPUT = (
    RESULTS_DIR
    / "xgboost_threshold_005_evaluation.json"
)

PNG_OUTPUT = (
    RESULTS_DIR
    / "xgboost_threshold_005_confusion_matrix.png"
)


# ============================================================
# CONFIGURATION
# ============================================================

THRESHOLD = 0.05

EXCLUDED_COLUMNS = {
    "Label",
    "canonical_label",
    "binary_label",
    "original_label",
    "attack_cat",
}


# ============================================================
# FIND FEATURE COLUMNS
# ============================================================

def get_feature_columns():

    files = sorted(VALIDATION_ROOT.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No validation CSV files found in {VALIDATION_ROOT}"
        )

    df = pd.read_csv(files[0], nrows=5)

    feature_columns = [
        column
        for column in df.columns
        if (
            column not in EXCLUDED_COLUMNS
            and pd.api.types.is_numeric_dtype(df[column])
        )
    ]

    print(f"Feature columns: {len(feature_columns)}")

    return feature_columns


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

def load_validation(feature_columns):

    X_parts = []
    y_parts = []

    files = sorted(VALIDATION_ROOT.glob("*.csv"))

    print()
    print("Loading validation data...")
    print("=" * 70)

    for index, file in enumerate(files, start=1):

        print(
            f"{index:02d}/{len(files)} "
            f"{file.name}"
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

        valid_rows = (
            X.notna().all(axis=1)
            & y.notna()
        )

        X = X.loc[valid_rows].astype(np.float32)
        y = y.loc[valid_rows].astype(np.int8)

        X_parts.append(X)
        y_parts.append(y)

    X_val = pd.concat(
        X_parts,
        ignore_index=True
    )

    y_val = pd.concat(
        y_parts,
        ignore_index=True
    )

    print("=" * 70)

    print(f"Validation rows: {len(X_val):,}")
    print(f"Validation features: {X_val.shape[1]}")

    return X_val, y_val


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("XGBOOST THRESHOLD EVALUATION")
    print("=" * 70)

    print(f"Threshold: {THRESHOLD}")
    print(f"Model: {MODEL_PATH}")

    # --------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------

    feature_columns = get_feature_columns()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print("Loading XGBoost model...")

    model = xgb.XGBClassifier()

    model.load_model(MODEL_PATH)

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    X_val, y_val = load_validation(
        feature_columns
    )

    # --------------------------------------------------------
    # Generate probabilities
    # --------------------------------------------------------

    print()
    print("Generating attack probabilities...")

    probabilities = model.predict_proba(
        X_val
    )[:, 1]

    # --------------------------------------------------------
    # Apply threshold
    # --------------------------------------------------------

    predictions = (
        probabilities >= THRESHOLD
    ).astype(np.int8)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    # --------------------------------------------------------
    # Security metrics
    # --------------------------------------------------------

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"Threshold          : {THRESHOLD:.2f}")

    print()
    print(f"Accuracy           : {accuracy:.6f}")
    print(f"Precision          : {precision:.6f}")
    print(f"Recall             : {recall:.6f}")
    print(f"F1 Score           : {f1:.6f}")
    print(f"ROC-AUC            : {roc_auc:.6f}")

    print()
    print("CONFUSION MATRIX")
    print("-" * 40)

    print("                 Predicted")
    print("                 Benign   Attack")
    print(
        f"Actual Benign    {tn:8,d} {fp:8,d}"
    )
    print(
        f"Actual Attack    {fn:8,d} {tp:8,d}"
    )

    print()
    print(f"True Negatives    : {tn:,}")
    print(f"False Positives   : {fp:,}")
    print(f"False Negatives   : {fn:,}")
    print(f"True Positives    : {tp:,}")

    print()
    print(f"False Positive Rate: {false_positive_rate:.6f}")
    print(f"False Negative Rate: {false_negative_rate:.6f}")

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_val,
        predictions,
        target_names=[
            "BENIGN",
            "ATTACK"
        ],
        output_dict=True,
        zero_division=0
    )

    print()
    print("CLASSIFICATION REPORT")
    print("-" * 70)

    print(
        classification_report(
            y_val,
            predictions,
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

    results = {
        "dataset": "CIC-IDS2017",
        "model": "XGBoost",
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
            "false_positive_rate": float(false_positive_rate),
            "false_negative_rate": float(false_negative_rate),
        },

        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },

        "classification_report": report,
    }

    with open(
        JSON_OUTPUT,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print()
    print(f"JSON saved:")
    print(JSON_OUTPUT)

    # --------------------------------------------------------
    # Save confusion matrix image
    # --------------------------------------------------------

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    image = ax.imshow(cm)

    ax.set_title(
        "CIC-IDS2017 XGBoost\n"
        "Confusion Matrix @ Threshold 0.05"
    )

    ax.set_xlabel(
        "Predicted Label"
    )

    ax.set_ylabel(
        "Actual Label"
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels([
        "BENIGN",
        "ATTACK"
    ])

    ax.set_yticklabels([
        "BENIGN",
        "ATTACK"
    ])

    for i in range(2):
        for j in range(2):

            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center"
            )

    fig.colorbar(
        image,
        ax=ax
    )

    plt.tight_layout()

    plt.savefig(
        PNG_OUTPUT,
        dpi=200
    )

    plt.close()

    print()
    print("Confusion matrix image saved:")
    print(PNG_OUTPUT)

    print()
    print("=" * 70)
    print("XGBOOST THRESHOLD EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()