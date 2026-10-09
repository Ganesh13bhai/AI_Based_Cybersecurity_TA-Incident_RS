from pathlib import Path
import json
import time

import joblib
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

SPLIT_ROOT = (
    ROOT
    / "data"
    / "splits"
    / "cicids2017"
)

MODEL_DIR = (
    ROOT
    / "models"
    / "cicids2017"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "ml"
    / "cicids2017"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "binary_label"

EXCLUDED_COLUMNS = {
    "Label",
    "canonical_label",
    "binary_label",
    "original_label",
    "attack_cat",
}

MAX_TRAIN_ROWS = 500_000

RANDOM_STATE = 42

CHUNK_SIZE = 100_000


# ============================================================
# XGBOOST CONFIGURATION
# ============================================================

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "learning_rate": 0.10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ============================================================
# FEATURE DISCOVERY
# ============================================================

def get_feature_columns():

    train_dir = SPLIT_ROOT / "train"

    first_file = sorted(
        train_dir.glob("*.csv")
    )[0]

    df = pd.read_csv(
        first_file,
        nrows=5
    )

    feature_columns = [
        column
        for column in df.columns
        if (
            column not in EXCLUDED_COLUMNS
            and pd.api.types.is_numeric_dtype(
                df[column]
            )
        )
    ]

    return feature_columns


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_training_data(feature_columns):

    train_dir = SPLIT_ROOT / "train"

    files = sorted(
        train_dir.glob("*.csv")
    )

    X_parts = []
    y_parts = []

    rows_loaded = 0

    print()
    print("Loading training data")
    print("=" * 60)

    for file in files:

        if rows_loaded >= MAX_TRAIN_ROWS:
            break

        remaining = (
            MAX_TRAIN_ROWS
            - rows_loaded
        )

        read_rows = min(
            CHUNK_SIZE,
            remaining
        )

        print(
            f"Reading {file.name} "
            f"(up to {read_rows:,} rows)"
        )

        df = pd.read_csv(
            file,
            usecols=feature_columns + [TARGET],
            nrows=read_rows
        )

        X = df[feature_columns].apply(
            pd.to_numeric,
            errors="coerce"
        )

        y = pd.to_numeric(
            df[TARGET],
            errors="coerce"
        )

        # Replace infinity.
        X = X.replace(
            [np.inf, -np.inf],
            np.nan
        )

        # Remove invalid rows.
        valid = (
            X.notna().all(axis=1)
            & y.notna()
        )

        X = X.loc[valid]
        y = y.loc[valid]

        # Reduce memory usage.
        X = X.astype(np.float32)
        y = y.astype(np.int8)

        X_parts.append(X)
        y_parts.append(y)

        rows_loaded += len(X)

        print(
            f"  Valid rows: {len(X):,}"
        )

    X_train = pd.concat(
        X_parts,
        ignore_index=True
    )

    y_train = pd.concat(
        y_parts,
        ignore_index=True
    )

    print()
    print("Training data loaded")
    print("-" * 60)

    print(
        "Rows:",
        f"{len(X_train):,}"
    )

    print(
        "Features:",
        len(X_train.columns)
    )

    print()
    print("Class distribution:")

    print(
        y_train
        .value_counts()
        .sort_index()
    )

    return X_train, y_train


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

def load_validation_data(feature_columns):

    validation_dir = (
        SPLIT_ROOT
        / "validation"
    )

    files = sorted(
        validation_dir.glob("*.csv")
    )

    X_parts = []
    y_parts = []

    print()
    print("Loading validation data")
    print("=" * 60)

    for file in files:

        print(
            f"Reading {file.name}"
        )

        df = pd.read_csv(
            file,
            usecols=feature_columns + [TARGET]
        )

        X = df[feature_columns].apply(
            pd.to_numeric,
            errors="coerce"
        )

        y = pd.to_numeric(
            df[TARGET],
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

        X = X.loc[valid].astype(
            np.float32
        )

        y = y.loc[valid].astype(
            np.int8
        )

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

    print()
    print("Validation data loaded")
    print("-" * 60)

    print(
        "Rows:",
        f"{len(X_val):,}"
    )

    print(
        "Features:",
        len(X_val.columns)
    )

    return X_val, y_val


# ============================================================
# MAIN TRAINING
# ============================================================

def train():

    print()
    print("=" * 60)
    print("CIC-IDS2017 XGBOOST")
    print("Binary Attack Detection")
    print("=" * 60)

    print()
    print("XGBoost version:", xgb.__version__)

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    feature_columns = (
        get_feature_columns()
    )

    print()
    print(
        "Number of features:",
        len(feature_columns)
    )

    # --------------------------------------------------------
    # Training data
    # --------------------------------------------------------

    X_train, y_train = (
        load_training_data(
            feature_columns
        )
    )

    # --------------------------------------------------------
    # Validation data
    # --------------------------------------------------------

    X_val, y_val = (
        load_validation_data(
            feature_columns
        )
    )

    # --------------------------------------------------------
    # Calculate imbalance ratio
    # --------------------------------------------------------

    negative_count = int(
        (y_train == 0).sum()
    )

    positive_count = int(
        (y_train == 1).sum()
    )

    scale_pos_weight = (
        negative_count
        / positive_count
    )

    print()
    print(
        "Negative samples:",
        f"{negative_count:,}"
    )

    print(
        "Positive samples:",
        f"{positive_count:,}"
    )

    print(
        "scale_pos_weight:",
        f"{scale_pos_weight:.4f}"
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    params = XGB_PARAMS.copy()

    params[
        "scale_pos_weight"
    ] = scale_pos_weight

    model = xgb.XGBClassifier(
        **params
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("TRAINING XGBOOST")
    print("=" * 60)

    start = time.perf_counter()

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_val, y_val)
        ],
        verbose=False
    )

    training_time = (
        time.perf_counter()
        - start
    )

    print()
    print(
        "Training completed in "
        f"{training_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # Validation prediction
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("VALIDATION")
    print("=" * 60)

    start = time.perf_counter()

    y_pred = model.predict(
        X_val
    )

    y_prob = model.predict_proba(
        X_val
    )[:, 1]

    prediction_time = (
        time.perf_counter()
        - start
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_val,
        y_pred
    )

    precision = precision_score(
        y_val,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_val,
        y_prob
    )

    cm = confusion_matrix(
        y_val,
        y_pred
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()

    print(
        f"Accuracy : {accuracy:.6f}"
    )

    print(
        f"Precision: {precision:.6f}"
    )

    print(
        f"Recall   : {recall:.6f}"
    )

    print(
        f"F1       : {f1:.6f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.6f}"
    )

    print()

    print("Confusion Matrix:")

    print(cm)

    print()

    print(
        "Classification Report:"
    )

    print(
        classification_report(
            y_val,
            y_pred,
            target_names=[
                "BENIGN",
                "ATTACK"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR
        / "xgboost_binary.json"
    )

    model.save_model(
        model_path
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = {

        "dataset": "CIC-IDS2017",

        "task":
            "binary_attack_detection",

        "model":
            "XGBClassifier",

        "xgboost_version":
            xgb.__version__,

        "features":
            len(feature_columns),

        "training_rows":
            int(len(X_train)),

        "validation_rows":
            int(len(X_val)),

        "n_estimators":
            XGB_PARAMS[
                "n_estimators"
            ],

        "max_depth":
            XGB_PARAMS[
                "max_depth"
            ],

        "learning_rate":
            XGB_PARAMS[
                "learning_rate"
            ],

        "subsample":
            XGB_PARAMS[
                "subsample"
            ],

        "colsample_bytree":
            XGB_PARAMS[
                "colsample_bytree"
            ],

        "scale_pos_weight":
            float(scale_pos_weight),

        "random_state":
            RANDOM_STATE,

        "training_time_seconds":
            float(training_time),

        "prediction_time_seconds":
            float(prediction_time),

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "roc_auc":
            float(roc_auc),

        "confusion_matrix":
            cm.tolist(),

        "feature_columns":
            feature_columns,
    }

    result_path = (
        RESULT_DIR
        / "xgboost_binary.json"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("SAVED")
    print("=" * 60)

    print(
        "Model:"
    )

    print(
        model_path
    )

    print()

    print(
        "Results:"
    )

    print(
        result_path
    )


if __name__ == "__main__":
    train()