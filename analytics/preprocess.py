"""
Module 2 - Titanic Data Preprocessing

Missing-value rules from the assignment:
- Less than 5% missing: drop affected rows.
- 5% to 30% missing: impute.
- More than 30% missing: explicitly justify dropping the column.

Titanic decisions:
- age: ~19.9% missing -> median imputation.
- embarked: ~0.2% missing -> drop affected rows.
- embark_town: ~0.2% missing -> drop affected rows.
- deck: ~77.2% missing -> drop the column because too much data is missing.
"""

from pathlib import Path

import pandas as pd


ANALYTICS_DIR = Path(__file__).resolve().parent
CSV_FILE = ANALYTICS_DIR / "titanic.csv"
OUTPUT_FILE = ANALYTICS_DIR / "titanic_cleaned.csv"


def calculate_missing_percentages(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate missing counts and percentages for every column."""

    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100

    result = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_percent": missing_percent.round(2),
        }
    )

    return result


def preprocess_titanic(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the assignment's missing-value rules."""

    df = df.copy()

    print("\nMissing values BEFORE preprocessing:")
    print(calculate_missing_percentages(df))

    # ---------------------------------------------------------
    # 1. age: 5%-30% missing -> median imputation
    # ---------------------------------------------------------
    age_median = df["age"].median()

    df["age"] = df["age"].fillna(age_median)

    print(f"\nAge median used for imputation: {age_median}")

    # ---------------------------------------------------------
    # 2. embarked: less than 5% missing -> drop rows
    # ---------------------------------------------------------
    df = df.dropna(subset=["embarked"])

    # ---------------------------------------------------------
    # 3. embark_town: less than 5% missing -> drop rows
    # ---------------------------------------------------------
    df = df.dropna(subset=["embark_town"])

    # ---------------------------------------------------------
    # 4. deck: more than 30% missing -> drop column
    # ---------------------------------------------------------
    # deck has approximately 77% missing values.
    # Keeping it would retain a column where most observations
    # have no value, so we explicitly drop it.
    df = df.drop(columns=["deck"])

    print("\nMissing values AFTER preprocessing:")
    print(calculate_missing_percentages(df))

    return df


def main() -> None:
    """Load the CSV and preprocess the Titanic dataset."""

    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"{CSV_FILE} was not found. "
            "Run titanic_data.py first."
        )

    # IMPORTANT:
    # From this point onward, we read the local CSV.
    # We do NOT call sns.load_dataset() again.
    df = pd.read_csv(CSV_FILE)

    print("Loaded Titanic dataset from CSV.")
    print(f"Original shape: {df.shape}")

    cleaned_df = preprocess_titanic(df)

    print(f"\nCleaned shape: {cleaned_df.shape}")

    # Save the cleaned dataset for later analysis.
    cleaned_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Cleaned dataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()