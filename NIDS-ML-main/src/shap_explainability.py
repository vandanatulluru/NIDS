"""
SHAP Explainability Module
Explains why the trained intrusion detection model makes predictions.
"""

from pathlib import Path
import warnings

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
TRAIN_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "selected_train_data.csv"
)

RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

# SHAP can become extremely expensive with a huge dataset.
# We use representative samples for explanations.
BACKGROUND_SAMPLES = 500
EXPLANATION_SAMPLES = 1000


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """Load the trained best model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found:\n{MODEL_PATH}\n\n"
            "Run model training first."
        )

    print("\nLoading trained model...")

    model = joblib.load(MODEL_PATH)

    print(
        f"Model loaded: {type(model).__name__}"
    )

    return model


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_training_data():
    """Load selected training data."""

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found:\n{TRAIN_PATH}\n\n"
            "Run preprocessing and feature selection first."
        )

    df = pd.read_csv(TRAIN_PATH)

    if "Label" not in df.columns:
        raise ValueError(
            "Label column not found in training data."
        )

    X = df.drop(columns=["Label"])

    # Remove any accidental non-numeric columns
    X = X.select_dtypes(include=[np.number])

    print(
        f"Training data loaded: {len(X):,} samples"
    )

    print(
        f"Features available: {len(X.columns)}"
    )

    return X


# ============================================================
# SAMPLE DATA
# ============================================================

def create_samples(X):
    """
    Create manageable background and explanation samples.
    """

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    # Background data
    if len(X) > BACKGROUND_SAMPLES:
        background_indices = rng.choice(
            len(X),
            size=BACKGROUND_SAMPLES,
            replace=False,
        )

        background = X.iloc[
            background_indices
        ].copy()

    else:
        background = X.copy()

    # Explanation data
    if len(X) > EXPLANATION_SAMPLES:
        explanation_indices = rng.choice(
            len(X),
            size=EXPLANATION_SAMPLES,
            replace=False,
        )

        explanation = X.iloc[
            explanation_indices
        ].copy()

    else:
        explanation = X.copy()

    return background, explanation


# ============================================================
# CREATE SHAP EXPLAINER
# ============================================================

def create_explainer(model, background):
    """
    Create an appropriate SHAP explainer.

    TreeExplainer is used for tree-based models such as
    Random Forest and XGBoost.
    """

    print("\nCreating SHAP explainer...")

    model_name = type(model).__name__

    if model_name in [
        "RandomForestClassifier",
        "XGBClassifier",
        "GradientBoostingClassifier",
        "DecisionTreeClassifier",
    ]:

        explainer = shap.TreeExplainer(
            model
        )

        print(
            "Using SHAP TreeExplainer."
        )

    else:

        # Generic fallback for other models
        explainer = shap.Explainer(
            model,
            background,
        )

        print(
            "Using generic SHAP Explainer."
        )

    return explainer


# ============================================================
# NORMALIZE SHAP VALUES
# ============================================================

def get_positive_class_shap_values(
    shap_values
):
    """
    Normalize SHAP output so that the returned matrix
    represents the ATTACK class.

    Different SHAP versions return different formats.
    """

    # Newer SHAP versions can return an Explanation object
    if isinstance(
        shap_values,
        shap.Explanation,
    ):

        values = shap_values.values

    else:

        values = shap_values

    # Some classifiers return:
    # [class_0_values, class_1_values]
    if isinstance(values, list):

        if len(values) == 2:
            return np.asarray(
                values[1]
            )

        return np.asarray(
            values[0]
        )

    values = np.asarray(values)

    # Shape:
    # samples x features x classes
    if values.ndim == 3:

        if values.shape[-1] == 2:
            return values[:, :, 1]

        return values[:, :, 0]

    return values


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

def calculate_feature_importance(
    shap_values,
    feature_names,
):
    """Calculate mean absolute SHAP importance."""

    importance = np.mean(
        np.abs(shap_values),
        axis=0,
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": importance,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            by="mean_abs_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return importance_df


# ============================================================
# SHAP BAR PLOT
# ============================================================

def create_bar_plot(
    shap_values,
    explanation_data,
    feature_names,
):
    """Create global SHAP feature importance plot."""

    print("\nCreating SHAP bar plot...")

    shap.summary_plot(
        shap_values,
        explanation_data,
        feature_names=feature_names,
        plot_type="bar",
        max_display=20,
        show=False,
    )

    plt.title(
        "Top 20 Features - SHAP Importance"
    )

    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / "shap_bar.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# SHAP SUMMARY PLOT
# ============================================================

def create_summary_plot(
    shap_values,
    explanation_data,
    feature_names,
):
    """Create detailed SHAP summary plot."""

    print(
        "\nCreating SHAP summary plot..."
    )

    shap.summary_plot(
        shap_values,
        explanation_data,
        feature_names=feature_names,
        max_display=20,
        show=False,
    )

    plt.title(
        "SHAP Summary - Feature Impact on Attack Detection"
    )

    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / "shap_summary.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# SAVE IMPORTANCE TABLE
# ============================================================

def save_importance_table(
    importance_df,
):
    """Save SHAP feature importance values."""

    output_path = (
        RESULTS_DIR
        / "shap_feature_importance.csv"
    )

    importance_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved: {output_path}"
    )

    print("\nTop 20 features:")

    print(
        importance_df.head(20)
        .to_string(index=False)
    )


# ============================================================
# EXPLAIN SINGLE SAMPLE
# ============================================================

def explain_single_prediction(
    model,
    explainer,
    X,
    sample_index=0,
):
    """
    Explain one network-flow prediction.
    """

    if sample_index >= len(X):
        raise IndexError(
            "Sample index is outside the dataset."
        )

    sample = X.iloc[
        [sample_index]
    ]

    prediction = model.predict(
        sample
    )[0]

    if hasattr(
        model,
        "predict_proba",
    ):

        attack_probability = (
            model.predict_proba(sample)[0][1]
        )

    else:

        attack_probability = None

    shap_result = explainer(
        sample
    )

    shap_values = get_positive_class_shap_values(
        shap_result
    )

    feature_contributions = pd.DataFrame(
        {
            "feature": X.columns,
            "shap_value": shap_values[0],
        }
    )

    feature_contributions[
        "absolute_shap"
    ] = np.abs(
        feature_contributions[
            "shap_value"
        ]
    )

    feature_contributions = (
        feature_contributions
        .sort_values(
            "absolute_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print("\n" + "=" * 60)
    print("SINGLE PREDICTION EXPLANATION")
    print("=" * 60)

    print(
        f"Prediction: "
        f"{'ATTACK' if prediction == 1 else 'BENIGN'}"
    )

    if attack_probability is not None:
        print(
            f"Attack probability: "
            f"{attack_probability:.4f}"
        )

    print("\nTop contributing features:")

    print(
        feature_contributions
        .head(10)
        .to_string(index=False)
    )

    return feature_contributions


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("NETWORK INTRUSION DETECTION")
    print("SHAP EXPLAINABILITY")
    print("=" * 60)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X = load_training_data()

    # --------------------------------------------------------
    # Create samples
    # --------------------------------------------------------

    background, explanation = create_samples(
        X
    )

    print(
        f"\nBackground samples: "
        f"{len(background):,}"
    )

    print(
        f"Explanation samples: "
        f"{len(explanation):,}"
    )

    # --------------------------------------------------------
    # Create explainer
    # --------------------------------------------------------

    explainer = create_explainer(
        model,
        background,
    )

    # --------------------------------------------------------
    # Calculate SHAP values
    # --------------------------------------------------------

    print(
        "\nCalculating SHAP values..."
    )

    shap_result = explainer(
        explanation
    )

    shap_values = (
        get_positive_class_shap_values(
            shap_result
        )
    )

    print(
        f"SHAP matrix shape: "
        f"{shap_values.shape}"
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance_df = (
        calculate_feature_importance(
            shap_values,
            explanation.columns,
        )
    )

    save_importance_table(
        importance_df
    )

    # --------------------------------------------------------
    # Create plots
    # --------------------------------------------------------

    create_bar_plot(
        shap_values,
        explanation,
        explanation.columns,
    )

    create_summary_plot(
        shap_values,
        explanation,
        explanation.columns,
    )

    # --------------------------------------------------------
    # Explain one sample
    # --------------------------------------------------------

    explain_single_prediction(
        model,
        explainer,
        explanation,
        sample_index=0,
    )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SHAP EXPLAINABILITY COMPLETED")
    print("=" * 60)

    print("\nGenerated files:")

    print(
        "  results/shap_feature_importance.csv"
    )

    print(
        "  results/plots/shap_bar.png"
    )

    print(
        "  results/plots/shap_summary.png"
    )


if __name__ == "__main__":
    main()