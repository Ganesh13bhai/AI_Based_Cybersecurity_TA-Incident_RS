from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
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

TRAIN_ROOT = ROOT / "data" / "splits" / "cicids2017" / "train"
VAL_ROOT = ROOT / "data" / "splits" / "cicids2017" / "validation"

MODEL_DIR = ROOT / "models" / "cicids2017"
RESULT_DIR = ROOT / "results" / "ml" / "cicids2017"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "extra_trees_binary.joblib"
RESULT_PATH = RESULT_DIR / "extra_trees_binary.json"


# ============================================================
# CONFIGURATION
# ============================================================

MAX_TRAIN_ROWS = 500_000
RANDOM_STATE = 42

EXCLUDED_COLUMNS = {
    "Label",
    "canonical_label",
    "binary_label",
    "original_label",
    "attack_cat",
}


# ============================================================
# GET FEATURE COLUMNS
# ============================================================

def get_feature_columns():

    files = sorted(VAL_ROOT.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No validation files found: {VAL_ROOT}"
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

    return feature_columns


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_training_data(feature_columns):

    X_parts = []
    y_parts = []

    total_loaded = 0

    files = sorted(TRAIN_ROOT.glob("*.csv"))

    print()
    print("Loading training data...")
    print("=" * 70)

    for index, file in enumerate(files, start=1):

        if total_loaded >= MAX_TRAIN_ROWS:
            break

        remaining = MAX_TRAIN_ROWS - total_loaded

        df = pd.read_csv(
            file,
            usecols=feature_columns + ["binary_label"],
            nrows=remaining,
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

        X = X.loc[valid].astype(np.float32)
        y = y.loc[valid].astype(np.int8)

        X_parts.append(X)
        y_parts.append(y)

        total_loaded += len(X)

        print(
            f"{index:02d}/{len(files)} "
            f"{file.name}: "
            f"{len(X):,} rows"
        )

    X_train = pd.concat(
        X_parts,
        ignore_index=True
    )

    y_train = pd.concat(
        y_parts,
        ignore_index=True
    )

    print("=" * 70)
    print(f"Training rows: {len(X_train):,}")
    print(f"Features: {X_train.shape[1]}")

    return X_train, y_train


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

def load_validation_data(feature_columns):

    X_parts = []
    y_parts = []

    files = sorted(VAL_ROOT.glob("*.csv"))

    print()
    print("Loading validation data...")
    print("=" * 70)

    for index, file in enumerate(files, start=1):

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

        X = X.loc[valid].astype(np.float32)
        y = y.loc[valid].astype(np.int8)

        X_parts.append(X)
        y_parts.append(y)

        print(
            f"{index:02d}/{len(files)} "
            f"{file.name}: "
            f"{len(X):,} rows"
        )

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

    return X_val, y_val


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("CIC-IDS2017 EXTRA TREES BINARY CLASSIFICATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    feature_columns = get_feature_columns()

    print()
    print(f"Number of features: {len(feature_columns)}")

    # --------------------------------------------------------
    # Training data
    # --------------------------------------------------------

    X_train, y_train = load_training_data(
        feature_columns
    )

    print()
    print("Training class distribution:")

    print(
        y_train.value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Validation data
    # --------------------------------------------------------

    X_val, y_val = load_validation_data(
        feature_columns
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("Creating Extra Trees model...")

    model = ExtraTreesClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
        max_features="sqrt",
        min_samples_leaf=1,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print()
    print("Training Extra Trees...")

    start_time = time.time()

    model.fit(
        X_train,
        y_train
    )

    training_time = time.time() - start_time

    print(
        f"Training completed in "
        f"{training_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # Validation prediction
    # --------------------------------------------------------

    print()
    print("Generating validation predictions...")

    prediction_start = time.time()

    probabilities = model.predict_proba(
        X_val
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(np.int8)

    prediction_time = time.time() - prediction_start

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

    cm = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

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
    print("EXTRA TREES VALIDATION RESULTS")
    print("=" * 70)

    print(f"Accuracy            : {accuracy:.6f}")
    print(f"Precision           : {precision:.6f}")
    print(f"Recall              : {recall:.6f}")
    print(f"F1 Score            : {f1:.6f}")
    print(f"ROC-AUC             : {roc_auc:.6f}")

    print()
    print("Confusion Matrix")
    print("-" * 50)

    print("                 Predicted")
    print("                 Benign     Attack")

    print(
        f"Actual Benign    {tn:8,d} {fp:10,d}"
    )

    print(
        f"Actual Attack    {fn:8,d} {tp:10,d}"
    )

    print()
    print(f"True Negatives     : {tn:,}")
    print(f"False Positives    : {fp:,}")
    print(f"False Negatives    : {fn:,}")
    print(f"True Positives     : {tp:,}")

    print()
    print(
        f"False Positive Rate: "
        f"{false_positive_rate:.6f}"
    )

    print(
        f"False Negative Rate: "
        f"{false_negative_rate:.6f}"
    )

    print()
    print("Classification Report")
    print("-" * 70)

    report_text = classification_report(
        y_val,
        predictions,
        target_names=[
            "BENIGN",
            "ATTACK"
        ],
        zero_division=0
    )

    print(report_text)

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    import joblib

    joblib.dump(
        model,
        MODEL_PATH
    )

    print()
    print(f"Model saved:")
    print(MODEL_PATH)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    report_dict = classification_report(
        y_val,
        predictions,
        target_names=[
            "BENIGN",
            "ATTACK"
        ],
        output_dict=True,
        zero_division=0
    )

    results = {
        "dataset": "CIC-IDS2017",
        "model": "ExtraTrees",
        "task": "binary_detection",

        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "feature_count": int(len(feature_columns)),

        "parameters": {
            "n_estimators": 200,
            "random_state": RANDOM_STATE,
            "class_weight": "balanced",
            "max_features": "sqrt",
            "min_samples_leaf": 1,
        },

        "threshold": 0.50,

        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "false_positive_rate": float(
                false_positive_rate
            ),
            "false_negative_rate": float(
                false_negative_rate
            ),
        },

        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },

        "timing": {
            "training_seconds": float(
                training_time
            ),
            "prediction_seconds": float(
                prediction_time
            ),
        },

        "classification_report": report_dict,
    }

    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print()
    print("Results saved:")
    print(RESULT_PATH)

    print()
    print("=" * 70)
    print("EXTRA TREES TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()