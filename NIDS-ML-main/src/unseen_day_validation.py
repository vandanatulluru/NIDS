import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RAW_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "best_model.pkl"
)

SCALER_FILE = os.path.join(
    PROCESSED_DIR,
    "scaler.pkl"
)

FEATURE_FILE = os.path.join(
    MODEL_DIR,
    "feature_names.txt"
)

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "unseen_day_validation.csv"
)

REPORT_FILE = os.path.join(
    RESULTS_DIR,
    "UNSEEN_DAY_VALIDATION_REPORT.txt"
)


# ============================================================
# CIC-IDS2017 CAPTURE FILES
# ============================================================

DATASET_FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv"
]


# ============================================================
# FEATURE LOADING
# ============================================================

def load_features():

    with open(
        FEATURE_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        features = [
            line.strip()
            for line in f
            if line.strip()
        ]

    print(
        "\nSelected features:",
        len(features)
    )

    return features


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\nLoading trained binary model...")

    model = joblib.load(
        MODEL_FILE
    )

    scaler = joblib.load(
        SCALER_FILE
    )

    print("✓ Model loaded")
    print("✓ Scaler loaded")

    return model, scaler


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

def normalize_columns(df):

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace(
            " ",
            "_",
            regex=False
        )
    )

    return df


# ============================================================
# LOAD ONE CAPTURE FILE
# ============================================================

def load_capture(
    filename,
    features
):

    filepath = os.path.join(
        RAW_DIR,
        filename
    )

    if not os.path.exists(filepath):

        raise FileNotFoundError(
            f"Dataset file not found:\n{filepath}"
        )

    print(
        f"\nLoading unseen capture:"
    )

    print(
        filename
    )

    df = pd.read_csv(
        filepath,
        low_memory=False
    )

    df = normalize_columns(df)

    print(
        "Original rows:",
        len(df)
    )

    if "Label" not in df.columns:

        raise ValueError(
            "Label column not found."
        )

    # --------------------------------------------------------
    # Clean labels
    # --------------------------------------------------------

    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates()

    print(
        "Duplicates removed:",
        before - len(df)
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    missing = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing features:\n"
            + "\n".join(missing)
        )

    X = df[features].copy()

    y = df["Label"].copy()

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    print(
        "Cleaned rows:",
        len(X)
    )

    print(
        "\nLabel distribution:"
    )

    print(
        y.value_counts()
    )

    return X, y


# ============================================================
# BINARY EVALUATION
# ============================================================

def evaluate(
    X,
    y,
    model,
    scaler,
    filename
):

    print("\n")
    print("=" * 70)
    print("UNSEEN CAPTURE EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Convert original labels to binary
    # --------------------------------------------------------

    y_true = (
        y.str.upper()
        .ne("BENIGN")
        .astype(int)
        .values
    )

    # --------------------------------------------------------
    # Scale using training scaler
    # --------------------------------------------------------

    X_scaled = scaler.transform(
        X
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = model.predict(
        X_scaled
    )

    probabilities = (
        model.predict_proba(
            X_scaled
        )[:, 1]
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    # ROC-AUC requires both classes
    if len(np.unique(y_true)) == 2:

        roc_auc = roc_auc_score(
            y_true,
            probabilities
        )

    else:

        roc_auc = float("nan")

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        "\nCapture:",
        filename
    )

    print(
        "\nAccuracy :",
        f"{accuracy:.4f}"
    )

    print(
        "Precision:",
        f"{precision:.4f}"
    )

    print(
        "Recall   :",
        f"{recall:.4f}"
    )

    print(
        "F1 Score :",
        f"{f1:.4f}"
    )

    if not np.isnan(roc_auc):

        print(
            "ROC-AUC  :",
            f"{roc_auc:.4f}"
        )

    print(
        "\nConfusion Matrix:"
    )

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    print(cm)

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=[
                "BENIGN",
                "ATTACK"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Detailed results
    # --------------------------------------------------------

    results = pd.DataFrame({

        "Capture": filename,

        "Ground_Truth": y.values,

        "Binary_Ground_Truth": np.where(
            y_true == 1,
            "ATTACK",
            "BENIGN"
        ),

        "Prediction": np.where(
            predictions == 1,
            "ATTACK",
            "BENIGN"
        ),

        "Attack_Probability": probabilities

    })

    results["Correct"] = (
        results["Binary_Ground_Truth"]
        ==
        results["Prediction"]
    )

    return (
        results,
        {
            "capture": filename,
            "rows": len(X),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm
        }
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("UNSEEN-DAY NETWORK INTRUSION VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    features = load_features()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model, scaler = load_model()

    # --------------------------------------------------------
    # We deliberately select ONE capture file that was not
    # used as a random row-level test set.
    #
    # Wednesday contains a large BENIGN + attack capture and
    # provides a useful unseen-capture experiment.
    # --------------------------------------------------------

    unseen_file = (
        "Wednesday-workingHours.pcap_ISCX.csv"
    )

    print(
        "\nUnseen capture selected:"
    )

    print(
        unseen_file
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    X, y = load_capture(
        unseen_file,
        features
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    results, metrics = evaluate(
        X,
        y,
        model,
        scaler,
        unseen_file
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\nDetailed results saved:"
    )

    print(
        OUTPUT_FILE
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report = []

    report.append(
        "=" * 70
    )

    report.append(
        "UNSEEN-DAY VALIDATION REPORT"
    )

    report.append(
        "=" * 70
    )

    report.append(
        f"\nCapture: {metrics['capture']}"
    )

    report.append(
        f"Rows: {metrics['rows']}"
    )

    report.append(
        "\nBinary Detection Performance"
    )

    report.append(
        "-" * 50
    )

    report.append(
        f"Accuracy : {metrics['accuracy']:.4f}"
    )

    report.append(
        f"Precision: {metrics['precision']:.4f}"
    )

    report.append(
        f"Recall   : {metrics['recall']:.4f}"
    )

    report.append(
        f"F1 Score : {metrics['f1']:.4f}"
    )

    if not np.isnan(
        metrics["roc_auc"]
    ):

        report.append(
            f"ROC-AUC  : {metrics['roc_auc']:.4f}"
        )

    report.append(
        "\nConfusion Matrix:"
    )

    report.append(
        str(
            metrics["confusion_matrix"]
        )
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(report)
        )

    print(
        "\nReport saved:"
    )

    print(
        REPORT_FILE
    )

    print("\n")
    print("=" * 70)
    print("UNSEEN-DAY VALIDATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()