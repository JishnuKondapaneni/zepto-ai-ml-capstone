"""
Save and verify the best-performing Titanic classification pipeline.

The selected model is Logistic Regression with class_weight="balanced".
The saved pipeline contains preprocessing + estimator so raw input can
be passed directly after reloading.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "titanic_cleaned.csv"
MODEL_PATH = BASE_DIR / "best_classification_pipeline.joblib"


def create_preprocessor() -> ColumnTransformer:
    """Create the same leakage-free preprocessing used for classification."""

    numeric_features = [
        "age",
        "sibsp",
        "parch",
        "fare",
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
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def main() -> None:
    """Train, save, reload, and verify the complete pipeline."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Required dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    # Remove direct target leakage.
    X = df.drop(columns=["survived", "alive"])
    y = df["survived"]

    # Use the same stratified split used by classification.py.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    preprocessor = create_preprocessor()

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    print("Training best classification pipeline...")
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)

    print(f"Saved model: {MODEL_PATH}")

    # Reload the complete pipeline.
    loaded_model = joblib.load(MODEL_PATH)

    # Verify that the reloaded pipeline accepts RAW input.
    raw_sample = X_test.head(1)
    prediction = loaded_model.predict(raw_sample)[0]
    probability = loaded_model.predict_proba(raw_sample)[0, 1]

    print("\nReload verification successful.")
    print(f"Raw input columns: {list(raw_sample.columns)}")
    print(f"Prediction: {prediction}")
    print(f"Survival probability: {probability:.4f}")

    if prediction not in [0, 1]:
        raise ValueError("Unexpected classification prediction.")

    print("\nCLASSIFICATION PIPELINE VERIFICATION PASSED")


if __name__ == "__main__":
    main()