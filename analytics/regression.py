"""
Module 2 - Titanic Fare Regression

Requirements:
- Predict fare using multivariate linear regression.
- Use leakage-free preprocessing.
- Report:
    - MAE
    - RMSE
    - R-squared
    - Adjusted R-squared
- Create a residual heteroscedasticity plot.
"""

from pathlib import Path

import matplotlib

# Use a non-GUI backend to avoid Tkinter errors on Windows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# FILE PATHS
# ============================================================

ANALYTICS_DIR = Path(__file__).resolve().parent

CSV_FILE = ANALYTICS_DIR / "titanic_cleaned.csv"

PLOTS_DIR = ANALYTICS_DIR / "plots"

PLOTS_DIR.mkdir(
    exist_ok=True
)

TARGET = "fare"


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor() -> ColumnTransformer:
    """
    Create preprocessing for the regression model.

    Numeric columns:
        - median imputation
        - standardization

    Categorical columns:
        - most-frequent imputation
        - one-hot encoding
    """

    numeric_features = [
        "survived",
        "pclass",
        "age",
        "sibsp",
        "parch",
    ]

    categorical_features = [
        "sex",
        "embarked",
        "class",
        "who",
        "adult_male",
        "embark_town",
        "alone",
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    return preprocessor


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Run the complete fare regression analysis."""

    # ========================================================
    # CHECK INPUT FILE
    # ========================================================

    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"{CSV_FILE} was not found. "
            "Run titanic_data.py and preprocess.py first."
        )

    # ========================================================
    # LOAD CLEANED DATA
    # ========================================================

    df = pd.read_csv(
        CSV_FILE
    )

    print(
        "Titanic cleaned dataset loaded."
    )

    print(
        f"Shape: {df.shape}"
    )

    # ========================================================
    # PREPARE FEATURES AND TARGET
    # ========================================================

    X = df.drop(
        columns=[TARGET, "alive"]
    )

    y = df[TARGET]

    print(
        f"\nTarget: {TARGET}"
    )

    print(
        f"Features: {X.shape[1]}"
    )

    print(
        f"Rows: {len(X)}"
    )

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
        )
    )

    print(
        "\nTRAIN/TEST SPLIT"
    )

    print(
        f"Training rows: {len(X_train)}"
    )

    print(
        f"Testing rows:  {len(X_test)}"
    )

    # ========================================================
    # BUILD LEAKAGE-FREE PIPELINE
    # ========================================================

    regression_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "regressor",
                LinearRegression(),
            ),
        ]
    )

    # ========================================================
    # FIT ONLY ON TRAINING DATA
    # ========================================================

    regression_pipeline.fit(
        X_train,
        y_train,
    )

    print(
        "\nRegression model trained successfully."
    )

    # ========================================================
    # PREDICTIONS
    # ========================================================

    predictions = regression_pipeline.predict(
        X_test
    )

    # ========================================================
    # REGRESSION METRICS
    # ========================================================

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    # --------------------------------------------------------
    # Adjusted R-squared
    #
    # n = number of test observations
    # p = number of model predictors after preprocessing
    # --------------------------------------------------------

    preprocessor = (
        regression_pipeline.named_steps[
            "preprocessor"
        ]
    )

    transformed_X_test = (
        preprocessor.transform(
            X_test
        )
    )

    n = len(y_test)

    p = transformed_X_test.shape[1]

    adjusted_r2 = (
        1
        - (
            (1 - r2)
            * (n - 1)
            / (n - p - 1)
        )
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "FARE REGRESSION RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        f"MAE:           {mae:.4f}"
    )

    print(
        f"RMSE:          {rmse:.4f}"
    )

    print(
        f"R-squared:     {r2:.4f}"
    )

    print(
        f"Adjusted R²:   {adjusted_r2:.4f}"
    )

    print(
        f"\nNumber of observations (n): {n}"
    )

    print(
        f"Number of predictors (p):   {p}"
    )

    # ========================================================
    # RESIDUALS
    # ========================================================

    residuals = (
        y_test.values
        - predictions
    )

    # ========================================================
    # RESIDUAL HETEROSCEDASTICITY PLOT
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.scatter(
        predictions,
        residuals,
        alpha=0.6,
    )

    plt.axhline(
        y=0,
        linestyle="--",
    )

    plt.xlabel(
        "Predicted Fare"
    )

    plt.ylabel(
        "Residuals"
    )

    plt.title(
        "Residuals vs Predicted Fare"
    )

    plt.tight_layout()

    residual_plot = (
        PLOTS_DIR
        / "fare_residuals_heteroscedasticity.png"
    )

    plt.savefig(
        residual_plot,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        "all"
    )

    print(
        f"\nSaved residual plot: "
        f"{residual_plot}"
    )

    # ========================================================
    # SAVE REGRESSION RESULTS
    # ========================================================

    results = pd.DataFrame(
        [
            {
                "model": "Multivariate Linear Regression",
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "adjusted_r2": adjusted_r2,
                "observations": n,
                "predictors_after_preprocessing": p,
            }
        ]
    )

    results_file = (
        ANALYTICS_DIR
        / "regression_results.csv"
    )

    results.to_csv(
        results_file,
        index=False,
    )

    print(
        f"Saved regression results: "
        f"{results_file}"
    )

    # ========================================================
    # SAMPLE PREDICTIONS
    # ========================================================

    sample_predictions = pd.DataFrame(
        {
            "actual_fare": y_test.values[:10],
            "predicted_fare": predictions[:10],
            "residual": residuals[:10],
        }
    )

    print(
        "\nFirst 10 predictions:"
    )

    print(
        sample_predictions.to_string(
            index=False
        )
    )

    print(
        "\nFARE REGRESSION ANALYSIS COMPLETE."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()