"""
Module 2 - Save and Verify Fare Regression Model

Requirements:
- Save the complete preprocessing + regression pipeline with joblib.
- Verify the saved pipeline can accept raw input data.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# FILE PATHS
# ============================================================

ANALYTICS_DIR = Path(__file__).resolve().parent

CSV_FILE = ANALYTICS_DIR / "titanic_cleaned.csv"

MODEL_FILE = ANALYTICS_DIR / "fare_regression_pipeline.joblib"


# ============================================================
# TARGET
# ============================================================

TARGET = "fare"


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor() -> ColumnTransformer:
    """Create the preprocessing transformer."""

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

    return ColumnTransformer(
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


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Train, save, load, and verify the regression pipeline."""

    # --------------------------------------------------------
    # Check CSV
    # --------------------------------------------------------

    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"{CSV_FILE} was not found."
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = pd.read_csv(
        CSV_FILE
    )

    print(
        "Titanic cleaned dataset loaded."
    )

    print(
        f"Shape: {df.shape}"
    )

    # --------------------------------------------------------
    # Separate features and target
    # --------------------------------------------------------

    X = df.drop(
        columns=[
            TARGET,
            "alive",
        ]
    )

    y = df[TARGET]

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
        )
    )

    print(
        f"\nTraining rows: {len(X_train)}"
    )

    print(
        f"Testing rows:  {len(X_test)}"
    )

    # --------------------------------------------------------
    # Complete pipeline
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Train pipeline
    # --------------------------------------------------------

    regression_pipeline.fit(
        X_train,
        y_train,
    )

    print(
        "\nComplete regression pipeline trained."
    )

    # --------------------------------------------------------
    # Save pipeline
    # --------------------------------------------------------

    joblib.dump(
        regression_pipeline,
        MODEL_FILE,
    )

    print(
        f"Saved model pipeline to:"
    )

    print(
        MODEL_FILE
    )

    # --------------------------------------------------------
    # Load saved pipeline
    # --------------------------------------------------------

    loaded_pipeline = joblib.load(
        MODEL_FILE
    )

    print(
        "\nSaved pipeline loaded successfully."
    )

    # --------------------------------------------------------
    # Verify using RAW input data
    #
    # No manual preprocessing is performed here.
    # The loaded pipeline performs all preprocessing itself.
    # --------------------------------------------------------

    raw_sample = X_test.iloc[
        [0]
    ].copy()

    print(
        "\nRAW INPUT SAMPLE:"
    )

    print(
        raw_sample.to_string(
            index=False
        )
    )

    prediction = loaded_pipeline.predict(
        raw_sample
    )

    print(
        "\nPrediction from saved pipeline:"
    )

    print(
        f"Predicted fare: {prediction[0]:.4f}"
    )

    # --------------------------------------------------------
    # Verify prediction is valid
    # --------------------------------------------------------

    if len(prediction) != 1:
        raise RuntimeError(
            "Model verification failed: "
            "expected exactly one prediction."
        )

    if pd.isna(prediction[0]):
        raise RuntimeError(
            "Model verification failed: "
            "prediction is NaN."
        )

    print(
        "\nMODEL VERIFICATION PASSED."
    )

    print(
        "The saved pipeline accepts raw input data "
        "and produces a valid prediction."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()