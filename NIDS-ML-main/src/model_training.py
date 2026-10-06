"""
Model Training Module
Leakage-safe training using pre-split train/test datasets.
"""

from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "data" / "processed" / "selected_train_data.csv"
TEST_PATH = BASE_DIR / "data" / "processed" / "selected_test_data.csv"

MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

# SVM can be extremely slow on the complete CICIDS2017 dataset.
# Therefore, only SVM is trained on a maximum of 100,000 samples.
SVM_MAX_SAMPLES = 100_000


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load the already split and feature-selected datasets."""

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found:\n{TRAIN_PATH}\n\n"
            "Run preprocessing and feature selection first."
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test data not found:\n{TEST_PATH}\n\n"
            "Run preprocessing and feature selection first."
        )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    if "Label" not in train_df.columns:
        raise ValueError("Label column not found in training data.")

    if "Label" not in test_df.columns:
        raise ValueError("Label column not found in test data.")

    # Ensure train and test have identical feature columns
    train_features = [c for c in train_df.columns if c != "Label"]
    test_features = [c for c in test_df.columns if c != "Label"]

    if train_features != test_features:
        raise ValueError(
            "Training and testing feature columns do not match."
        )

    X_train = train_df[train_features]
    y_train = train_df["Label"]

    X_test = test_df[test_features]
    y_test = test_df["Label"]

    print("\n" + "=" * 60)
    print("DATA LOADED")
    print("=" * 60)

    print(f"Training samples : {len(X_train):,}")
    print(f"Testing samples  : {len(X_test):,}")
    print(f"Features         : {len(train_features)}")

    print("\nTraining class distribution:")
    print(y_train.value_counts())

    print("\nTesting class distribution:")
    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test, train_features


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def create_models():
    """Create the machine-learning models."""

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=None,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "SVM": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    return models


# ============================================================
# SVM SAMPLING
# ============================================================

def sample_for_svm(X, y, max_samples=SVM_MAX_SAMPLES):
    """
    Stratified sampling for SVM to prevent excessive training time.
    """

    if len(X) <= max_samples:
        return X, y

    print(
        f"\nSVM training dataset is large ({len(X):,} samples)."
    )
    print(
        f"Using stratified sample of {max_samples:,} samples for SVM."
    )

    sampled_indices = []

    rng = np.random.RandomState(RANDOM_STATE)

    classes = np.unique(y)

    for cls in classes:
        class_indices = np.where(y.to_numpy() == cls)[0]

        proportion = len(class_indices) / len(y)
        n_samples = max(
            1,
            int(round(max_samples * proportion))
        )

        n_samples = min(n_samples, len(class_indices))

        selected = rng.choice(
            class_indices,
            size=n_samples,
            replace=False,
        )

        sampled_indices.extend(selected.tolist())

    sampled_indices = np.array(sampled_indices)

    # If rounding caused slightly more samples than requested
    if len(sampled_indices) > max_samples:
        sampled_indices = rng.choice(
            sampled_indices,
            size=max_samples,
            replace=False,
        )

    rng.shuffle(sampled_indices)

    X_sample = X.iloc[sampled_indices]
    y_sample = y.iloc[sampled_indices]

    return X_sample, y_sample


# ============================================================
# TRAIN + EVALUATE
# ============================================================

def train_and_evaluate(model_name, model, X_train, y_train, X_test, y_test):
    """Train one model and calculate evaluation metrics."""

    print("\n" + "=" * 60)
    print(f"TRAINING: {model_name}")
    print("=" * 60)

    train_X = X_train
    train_y = y_train

    # SVM gets a stratified sample
    if model_name == "SVM":
        train_X, train_y = sample_for_svm(
            X_train,
            y_train
        )

    print(f"Training samples used: {len(train_X):,}")

    start_time = time.time()

    model.fit(train_X, train_y)

    training_time = time.time() - start_time

    print(
        f"Training completed in {training_time:.2f} seconds."
    )

    # Predictions
    y_pred = model.predict(X_test)

    # Probability scores for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    try:
        roc_auc = roc_auc_score(
            y_test,
            y_score,
        )
    except ValueError:
        roc_auc = 0.0

    metrics = {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "training_time_seconds": training_time,
        "training_samples": len(train_X),
    }

    # Print metrics
    print("\nResults:")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["BENIGN", "ATTACK"],
            zero_division=0,
        )
    )

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.colorbar()

    plt.xticks(
        [0, 1],
        ["BENIGN", "ATTACK"]
    )

    plt.yticks(
        [0, 1],
        ["BENIGN", "ATTACK"]
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")

    plt.savefig(
        PLOTS_DIR / f"confusion_matrix_{safe_name}.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # Save ROC curve
    fpr, tpr, _ = roc_curve(
        y_test,
        y_score,
    )

    plt.figure(figsize=(7, 6))
    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {roc_auc:.4f})",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / f"roc_curve_{safe_name}.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return model, metrics


# ============================================================
# MODEL COMPARISON
# ============================================================

def save_model_comparison(results):
    """Save model comparison table and plot."""

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="f1_score",
        ascending=False,
    )

    comparison_path = RESULTS_DIR / "model_comparison.csv"

    results_df.to_csv(
        comparison_path,
        index=False,
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df[
            [
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
            ]
        ].to_string(index=False)
    )

    # Plot
    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    ]

    plot_df = results_df.set_index("model")[metric_columns]

    ax = plot_df.plot(
        kind="bar",
        figsize=(12, 7),
    )

    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Model")

    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "model_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return results_df


# ============================================================
# SAVE BEST MODEL
# ============================================================

def save_best_model(
    best_model,
    best_metrics,
    feature_names,
):
    """Save the best model and its metadata."""

    model_path = MODEL_DIR / "best_model.pkl"

    joblib.dump(
        best_model,
        model_path,
    )

    # Save feature names
    feature_path = MODEL_DIR / "feature_names.txt"

    with open(
        feature_path,
        "w",
        encoding="utf-8",
    ) as f:
        for feature in feature_names:
            f.write(f"{feature}\n")

    # Save metadata
    metadata = {
        "model": best_metrics["model"],
        "accuracy": float(best_metrics["accuracy"]),
        "precision": float(best_metrics["precision"]),
        "recall": float(best_metrics["recall"]),
        "f1_score": float(best_metrics["f1_score"]),
        "roc_auc": float(best_metrics["roc_auc"]),
        "training_time_seconds": float(
            best_metrics["training_time_seconds"]
        ),
        "training_samples": int(
            best_metrics["training_samples"]
        ),
        "number_of_features": len(feature_names),
        "feature_names": feature_names,
    }

    metadata_path = MODEL_DIR / "best_model_metadata.json"

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4,
        )

    # Also save readable metadata
    readable_path = MODEL_DIR / "best_model_metadata.txt"

    with open(
        readable_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write("BEST MODEL\n")
        f.write("=" * 50 + "\n\n")

        f.write(
            f"Model: {best_metrics['model']}\n"
        )
        f.write(
            f"Accuracy: {best_metrics['accuracy']:.6f}\n"
        )
        f.write(
            f"Precision: {best_metrics['precision']:.6f}\n"
        )
        f.write(
            f"Recall: {best_metrics['recall']:.6f}\n"
        )
        f.write(
            f"F1 Score: {best_metrics['f1_score']:.6f}\n"
        )
        f.write(
            f"ROC-AUC: {best_metrics['roc_auc']:.6f}\n"
        )
        f.write(
            f"Training Samples: {best_metrics['training_samples']}\n"
        )
        f.write(
            f"Number of Features: {len(feature_names)}\n"
        )

    print("\n" + "=" * 60)
    print("BEST MODEL SAVED")
    print("=" * 60)

    print(f"Model: {best_metrics['model']}")
    print(f"F1 Score: {best_metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {best_metrics['roc_auc']:.4f}")

    print(f"\nSaved to:")
    print(model_path)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("NETWORK INTRUSION DETECTION")
    print("MODEL TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load already split data
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
    ) = load_data()

    # --------------------------------------------------------
    # Create models
    # --------------------------------------------------------

    models = create_models()

    results = []
    trained_models = {}

    # --------------------------------------------------------
    # Train every model
    # --------------------------------------------------------

    for model_name, model in models.items():

        try:

            trained_model, metrics = train_and_evaluate(
                model_name,
                model,
                X_train,
                y_train,
                X_test,
                y_test,
            )

            trained_models[model_name] = trained_model
            results.append(metrics)

        except Exception as e:

            print(
                f"\nERROR while training {model_name}:"
            )
            print(e)

    # --------------------------------------------------------
    # Check results
    # --------------------------------------------------------

    if not results:
        raise RuntimeError(
            "No models were successfully trained."
        )

    # --------------------------------------------------------
    # Compare models
    # --------------------------------------------------------

    results_df = save_model_comparison(
        results
    )

    # --------------------------------------------------------
    # Select best model based on F1 score
    # --------------------------------------------------------

    best_row = results_df.iloc[0]

    best_model_name = best_row["model"]

    best_model = trained_models[
        best_model_name
    ]

    best_metrics = best_row.to_dict()

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    save_best_model(
        best_model,
        best_metrics,
        feature_names,
    )

    # --------------------------------------------------------
    # Save complete metrics
    # --------------------------------------------------------

    metrics_path = RESULTS_DIR / "model_metrics.txt"

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "NETWORK INTRUSION DETECTION MODEL RESULTS\n"
        )
        f.write("=" * 60 + "\n\n")

        for _, row in results_df.iterrows():

            f.write(
                f"Model: {row['model']}\n"
            )

            f.write(
                f"Accuracy: {row['accuracy']:.6f}\n"
            )

            f.write(
                f"Precision: {row['precision']:.6f}\n"
            )

            f.write(
                f"Recall: {row['recall']:.6f}\n"
            )

            f.write(
                f"F1 Score: {row['f1_score']:.6f}\n"
            )

            f.write(
                f"ROC-AUC: {row['roc_auc']:.6f}\n"
            )

            f.write(
                f"Training Time: "
                f"{row['training_time_seconds']:.2f} seconds\n"
            )

            f.write(
                f"Training Samples: "
                f"{int(row['training_samples'])}\n"
            )

            f.write("\n")

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"\nBest Model: {best_model_name}"
    )

    print(
        f"Best F1 Score: "
        f"{best_metrics['f1_score']:.4f}"
    )

    print(
        f"Best ROC-AUC: "
        f"{best_metrics['roc_auc']:.4f}"
    )

    print("\nGenerated files:")
    print("  models/best_model.pkl")
    print("  models/best_model_metadata.json")
    print("  models/best_model_metadata.txt")
    print("  models/feature_names.txt")
    print("  results/model_metrics.txt")
    print("  results/model_comparison.csv")
    print("  results/plots/")


if __name__ == "__main__":
    main()