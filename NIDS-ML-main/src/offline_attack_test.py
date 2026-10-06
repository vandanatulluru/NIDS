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
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

RAW_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

# Binary test data
BINARY_TEST_FILE = os.path.join(
    PROCESSED_DIR,
    "selected_test_data.csv"
)

# Models
BINARY_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "best_model.pkl"
)

BINARY_SCALER_FILE = os.path.join(
    PROCESSED_DIR,
    "scaler.pkl"
)

MULTICLASS_MODEL_FILE = os.path.join(
    MODEL_DIR,
    "attack_classifier.pkl"
)

MULTICLASS_SCALER_FILE = os.path.join(
    MODEL_DIR,
    "multiclass_scaler.pkl"
)

MULTICLASS_ENCODER_FILE = os.path.join(
    MODEL_DIR,
    "attack_label_encoder.pkl"
)

FEATURE_FILE = os.path.join(
    MODEL_DIR,
    "feature_names.txt"
)

# Output files
BINARY_RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "final_binary_validation.csv"
)

MULTICLASS_RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "final_multiclass_validation.csv"
)

FINAL_REPORT_FILE = os.path.join(
    RESULTS_DIR,
    "FINAL_VALIDATION_REPORT.txt"
)

MULTICLASS_CM_FILE = os.path.join(
    RESULTS_DIR,
    "final_multiclass_confusion_matrix.csv"
)


# ============================================================
# RAW CIC-IDS2017 FILES
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
# LOAD FEATURES
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
# LOAD MODELS
# ============================================================

def load_models():

    print("\nLoading models...")

    binary_model = joblib.load(
        BINARY_MODEL_FILE
    )

    binary_scaler = joblib.load(
        BINARY_SCALER_FILE
    )

    multiclass_model = joblib.load(
        MULTICLASS_MODEL_FILE
    )

    multiclass_scaler = joblib.load(
        MULTICLASS_SCALER_FILE
    )

    multiclass_encoder = joblib.load(
        MULTICLASS_ENCODER_FILE
    )

    print("✓ Binary model loaded")
    print("✓ Binary scaler loaded")
    print("✓ Multiclass model loaded")
    print("✓ Multiclass scaler loaded")
    print("✓ Label encoder loaded")

    return (
        binary_model,
        binary_scaler,
        multiclass_model,
        multiclass_scaler,
        multiclass_encoder
    )


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def normalize_columns(df):

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace(" ", "_", regex=False)
    )

    return df


# ============================================================
# BINARY VALIDATION
#
# IMPORTANT:
# selected_test_data.csv already contains the binary labels
# used by the binary pipeline.
# ============================================================

def evaluate_binary_model(
    features,
    binary_model,
    binary_scaler
):

    print("\n")
    print("=" * 70)
    print("FINAL BINARY MODEL VALIDATION")
    print("=" * 70)

    print(
        "\nLoading:",
        BINARY_TEST_FILE
    )

    df = pd.read_csv(
        BINARY_TEST_FILE
    )

    df = normalize_columns(df)

    print(
        "Test rows:",
        len(df)
    )

    # --------------------------------------------------------
    # Find label column
    # --------------------------------------------------------

    label_candidates = [
        "Label",
        "label",
        "target",
        "Target"
    ]

    label_col = None

    for col in label_candidates:

        if col in df.columns:

            label_col = col
            break

    if label_col is None:

        raise ValueError(
            "Binary label column not found."
        )

    print(
        "Binary label column:",
        label_col
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
            "Missing binary features:\n"
            + "\n".join(missing)
        )

    X = df[features].copy()

    # --------------------------------------------------------
    # Binary labels
    #
    # The preprocessing pipeline created:
    # BENIGN = 0
    # ATTACK = 1
    # --------------------------------------------------------

    y = pd.to_numeric(
        df[label_col],
        errors="coerce"
    )

    valid = y.notna()

    X = X.loc[valid].copy()
    y = y.loc[valid].astype(int)

    # --------------------------------------------------------
    # Clean numeric data
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

    # --------------------------------------------------------
    # Scale using EXACT training scaler
    # --------------------------------------------------------

    X_scaled = binary_scaler.transform(
        X
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = binary_model.predict(
        X_scaled
    )

    probabilities = binary_model.predict_proba(
        X_scaled
    )[:, 1]

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    print("\nBinary Results")
    print("-" * 50)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print("\nConfusion Matrix:")

    cm = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    )

    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y,
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
    # Save detailed results
    # --------------------------------------------------------

    binary_results = pd.DataFrame({

        "Ground_Truth": np.where(
            y.values == 1,
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

    binary_results["Correct"] = (
        binary_results["Ground_Truth"]
        ==
        binary_results["Prediction"]
    )

    binary_results.to_csv(
        BINARY_RESULTS_FILE,
        index=False
    )

    print(
        "\nBinary detailed results saved:"
    )

    print(
        BINARY_RESULTS_FILE
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm
    }


# ============================================================
# LOAD ORIGINAL CIC-IDS2017
#
# This is ONLY used for preserving the original attack names.
# ============================================================

def load_original_dataset(
    features
):

    print("\n")
    print("=" * 70)
    print("LOADING ORIGINAL CIC-IDS2017 LABELS")
    print("=" * 70)

    frames = []

    for filename in DATASET_FILES:

        path = os.path.join(
            RAW_DIR,
            filename
        )

        if not os.path.exists(path):

            print(
                f"WARNING: Missing {filename}"
            )

            continue

        print(
            f"Loading: {filename}"
        )

        df = pd.read_csv(
            path,
            low_memory=False
        )

        df = normalize_columns(df)

        frames.append(df)

    if not frames:

        raise FileNotFoundError(
            "No CIC-IDS2017 files found."
        )

    df = pd.concat(
        frames,
        ignore_index=True
    )

    print(
        "\nTotal raw rows:",
        len(df)
    )

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    if "Label" not in df.columns:

        raise ValueError(
            "Original Label column not found."
        )

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
            "Missing features in raw dataset:\n"
            + "\n".join(missing)
        )

    X = df[features].copy()

    y = df["Label"].copy()

    # --------------------------------------------------------
    # Numeric cleaning
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

    return X, y


# ============================================================
# MULTICLASS VALIDATION
# ============================================================

def evaluate_multiclass_model(
    X,
    y,
    multiclass_model,
    multiclass_scaler,
    multiclass_encoder
):

    print("\n")
    print("=" * 70)
    print("FINAL MULTICLASS ATTACK VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Only actual attacks
    # --------------------------------------------------------

    attack_mask = (
        y.str.upper()
        != "BENIGN"
    )

    X_attack = X.loc[
        attack_mask
    ].copy()

    y_attack = y.loc[
        attack_mask
    ].copy()

    print(
        "\nAttack samples:",
        len(X_attack)
    )

    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------

    X_scaled = multiclass_scaler.transform(
        X_attack
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    encoded_predictions = (
        multiclass_model.predict(
            X_scaled
        )
    )

    predicted_labels = (
        multiclass_encoder.inverse_transform(
            encoded_predictions.astype(int)
        )
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_attack,
        predicted_labels
    )

    macro_precision = precision_score(
        y_attack,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_attack,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_attack,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_attack,
        predicted_labels,
        average="weighted",
        zero_division=0
    )

    print("\nMulticlass Results")
    print("-" * 50)

    print(
        f"Accuracy       : {accuracy:.4f}"
    )

    print(
        f"Macro Precision: {macro_precision:.4f}"
    )

    print(
        f"Macro Recall   : {macro_recall:.4f}"
    )

    print(
        f"Macro F1       : {macro_f1:.4f}"
    )

    print(
        f"Weighted F1    : {weighted_f1:.4f}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print(
        "\nPer-Attack Classification Report:"
    )

    report = classification_report(
        y_attack,
        predicted_labels,
        zero_division=0
    )

    print(report)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    classes = multiclass_encoder.classes_

    cm = confusion_matrix(
        y_attack,
        predicted_labels,
        labels=classes
    )

    cm_df = pd.DataFrame(
        cm,
        index=classes,
        columns=classes
    )

    cm_df.to_csv(
        MULTICLASS_CM_FILE
    )

    print(
        "\nMulticlass confusion matrix saved:"
    )

    print(
        MULTICLASS_CM_FILE
    )

    # --------------------------------------------------------
    # Detailed results
    # --------------------------------------------------------

    results = pd.DataFrame({

        "Ground_Truth": y_attack.values,

        "Predicted_Attack_Type":
            predicted_labels

    })

    results["Correct"] = (
        results["Ground_Truth"]
        ==
        results["Predicted_Attack_Type"]
    )

    results.to_csv(
        MULTICLASS_RESULTS_FILE,
        index=False
    )

    print(
        "\nMulticlass detailed results saved:"
    )

    print(
        MULTICLASS_RESULTS_FILE
    )

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report": report,
        "confusion_matrix": cm
    }


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

def show_sample_predictions(
    X,
    y,
    binary_model,
    binary_scaler,
    multiclass_model,
    multiclass_scaler,
    multiclass_encoder
):

    print("\n")
    print("=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)

    # Select 20 random samples
    sample_count = min(
        20,
        len(X)
    )

    sample_indices = np.random.RandomState(
        42
    ).choice(
        len(X),
        size=sample_count,
        replace=False
    )

    X_sample = X.iloc[
        sample_indices
    ]

    y_sample = y.iloc[
        sample_indices
    ]

    # --------------------------------------------------------
    # Binary prediction
    # --------------------------------------------------------

    X_binary_scaled = (
        binary_scaler.transform(
            X_sample
        )
    )

    binary_predictions = (
        binary_model.predict(
            X_binary_scaled
        )
    )

    probabilities = (
        binary_model.predict_proba(
            X_binary_scaled
        )[:, 1]
    )

    # --------------------------------------------------------
    # Multiclass only for binary ATTACK predictions
    # --------------------------------------------------------

    attack_positions = np.where(
        binary_predictions == 1
    )[0]

    multiclass_predictions = {}

    if len(attack_positions) > 0:

        X_pred_attack = X_sample.iloc[
            attack_positions
        ]

        X_attack_scaled = (
            multiclass_scaler.transform(
                X_pred_attack
            )
        )

        encoded = (
            multiclass_model.predict(
                X_attack_scaled
            )
        )

        labels = (
            multiclass_encoder.inverse_transform(
                encoded.astype(int)
            )
        )

        for position, label in zip(
            attack_positions,
            labels
        ):

            multiclass_predictions[
                position
            ] = label

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    for i in range(
        sample_count
    ):

        ground_truth = str(
            y_sample.iloc[i]
        )

        binary_prediction = (
            "ATTACK"
            if binary_predictions[i] == 1
            else "BENIGN"
        )

        probability = probabilities[i]

        if i in multiclass_predictions:

            predicted_attack_type = (
                multiclass_predictions[i]
            )

        else:

            predicted_attack_type = "BENIGN"

        # Correct final prediction
        if ground_truth.upper() == "BENIGN":
            final_correct = (
                binary_predictions[i] == 0
            )
        else:
            final_correct = (
                binary_predictions[i] == 1
            and
            ground_truth.upper()
            == predicted_attack_type.upper()
        )

        print(
            "\n----------------------------------------"
        )

        print(
            "Ground Truth       :",
            ground_truth
        )

        print(
            "Binary Prediction  :",
            binary_prediction
        )

        print(
            "Attack Type        :",
            predicted_attack_type
        )

        print(
            "Attack Probability :",
            f"{probability * 100:.2f}%"
        )

        print(
            "Result             :",
            "✓ CORRECT"
            if final_correct
            else "✗ WRONG"
        )


# ============================================================
# FINAL REPORT
# ============================================================

def save_final_report(
    binary_results,
    multiclass_results
):

    report_lines = []

    report_lines.append(
        "=" * 70
    )

    report_lines.append(
        "FINAL NIDS MODEL VALIDATION REPORT"
    )

    report_lines.append(
        "=" * 70
    )

    report_lines.append(
        "\nBINARY ATTACK DETECTION"
    )

    report_lines.append(
        "-" * 50
    )

    report_lines.append(
        f"Accuracy : {binary_results['accuracy']:.4f}"
    )

    report_lines.append(
        f"Precision: {binary_results['precision']:.4f}"
    )

    report_lines.append(
        f"Recall   : {binary_results['recall']:.4f}"
    )

    report_lines.append(
        f"F1 Score : {binary_results['f1']:.4f}"
    )

    report_lines.append(
        f"ROC-AUC  : {binary_results['roc_auc']:.4f}"
    )

    report_lines.append(
        "\nConfusion Matrix:"
    )

    report_lines.append(
        str(binary_results["confusion_matrix"])
    )

    report_lines.append(
        "\n\nMULTICLASS ATTACK CLASSIFICATION"
    )

    report_lines.append(
        "-" * 50
    )

    report_lines.append(
        f"Accuracy       : {multiclass_results['accuracy']:.4f}"
    )

    report_lines.append(
        f"Macro Precision: {multiclass_results['macro_precision']:.4f}"
    )

    report_lines.append(
        f"Macro Recall   : {multiclass_results['macro_recall']:.4f}"
    )

    report_lines.append(
        f"Macro F1       : {multiclass_results['macro_f1']:.4f}"
    )

    report_lines.append(
        f"Weighted F1    : {multiclass_results['weighted_f1']:.4f}"
    )

    report_lines.append(
        "\nPer-Attack Classification Report:"
    )

    report_lines.append(
        multiclass_results["report"]
    )

    report_lines.append(
        "\n" + "=" * 70
    )

    report_lines.append(
        "VALIDATION COMPLETED"
    )

    report_lines.append(
        "=" * 70
    )

    with open(
        FINAL_REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(report_lines)
        )

    print(
        "\nFinal report saved:"
    )

    print(
        FINAL_REPORT_FILE
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINAL NIDS OFFLINE VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load features
    # --------------------------------------------------------

    features = load_features()

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    (
        binary_model,
        binary_scaler,
        multiclass_model,
        multiclass_scaler,
        multiclass_encoder
    ) = load_models()

    # --------------------------------------------------------
    # 1. Exact binary validation
    # --------------------------------------------------------

    binary_results = evaluate_binary_model(
        features,
        binary_model,
        binary_scaler
    )

    # --------------------------------------------------------
    # 2. Original dataset for multiclass labels
    # --------------------------------------------------------

    X_raw, y_raw = load_original_dataset(
        features
    )

    # --------------------------------------------------------
    # 3. Multiclass validation
    #
    # IMPORTANT:
    # The multiclass model was trained using an 80/20
    # stratified split. Recreate that split here.
    # --------------------------------------------------------

    binary_target = (
        y_raw.str.upper()
        != "BENIGN"
    ).astype(int)

    _, X_multiclass_test, _, y_multiclass_test = (
        train_test_split(
            X_raw,
            y_raw,
            test_size=0.20,
            random_state=42,
            stratify=binary_target
        )
    )

    multiclass_results = evaluate_multiclass_model(
        X_multiclass_test,
        y_multiclass_test,
        multiclass_model,
        multiclass_scaler,
        multiclass_encoder
    )

    # --------------------------------------------------------
    # 4. Sample predictions
    # --------------------------------------------------------

    show_sample_predictions(
        X_multiclass_test,
        y_multiclass_test,
        binary_model,
        binary_scaler,
        multiclass_model,
        multiclass_scaler,
        multiclass_encoder
    )

    # --------------------------------------------------------
    # 5. Save final report
    # --------------------------------------------------------

    save_final_report(
        binary_results,
        multiclass_results
    )

    print("\n")
    print("=" * 70)
    print("FINAL VALIDATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()