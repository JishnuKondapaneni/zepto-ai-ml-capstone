"""
Module 2 - Titanic Dataset Loading

Requirement:
1. Load seaborn Titanic dataset once.
2. Immediately export it to analytics/titanic.csv.
3. Use the CSV as the fallback dataset for all downstream analysis.
"""

from pathlib import Path

import pandas as pd
import seaborn as sns


# File locations
ANALYTICS_DIR = Path(__file__).resolve().parent
CSV_FILE = ANALYTICS_DIR / "titanic.csv"


def load_and_export_titanic() -> None:
    """
    Load the Titanic dataset once from Seaborn and immediately
    export it to analytics/titanic.csv.
    """

    print("Loading Titanic dataset from Seaborn...")

    # IMPORTANT:
    # This is the only sns.load_dataset() call in the project.
    df = sns.load_dataset("titanic")

    # Initial profiling required for the Titanic data inspection.
    print("\nDataset info:")
    df.info()

    print("\nDescriptive statistics:")
    print(df.describe())

    print("\nDataset shape:")
    print(df.shape)

    # Immediately export the original dataset to CSV.
    df.to_csv(CSV_FILE, index=False)

    print(f"Titanic dataset exported to: {CSV_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())


def load_from_csv() -> pd.DataFrame:
    """
    Load the Titanic dataset from the exported CSV.

    All downstream analytics should use this function instead of
    calling sns.load_dataset() again.
    """

    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"{CSV_FILE} does not exist. "
            "Run load_and_export_titanic() first."
        )

    df = pd.read_csv(CSV_FILE)

    print("\nTitanic dataset loaded from CSV.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    return df


if __name__ == "__main__":
    # Step 1: Load from Seaborn once and export immediately.
    load_and_export_titanic()

    # Step 2: Verify that the CSV can be used as the fallback dataset.
    df = load_from_csv()

    print("\nCSV verification successful.")
    print("\nFirst 5 rows:")
    print(df.head())
