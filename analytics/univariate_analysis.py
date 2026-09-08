"""
Module 2 - Titanic Univariate Analysis

Requirements:
- Histogram and boxplot for age.
- Histogram and boxplot for fare.
- IQR outlier counts for age and fare.
- Mean, median, and mode of fare.
- Explain the positive skew of fare.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ANALYTICS_DIR = Path(__file__).resolve().parent
CSV_FILE = ANALYTICS_DIR / "titanic_cleaned.csv"
PLOTS_DIR = ANALYTICS_DIR / "plots"

PLOTS_DIR.mkdir(exist_ok=True)


def calculate_iqr_outliers(series: pd.Series) -> tuple[float, float, int]:
    """Calculate IQR, lower/upper bounds, and number of outliers."""

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    outlier_count = (
        (series < lower_bound) | (series > upper_bound)
    ).sum()

    return iqr, lower_bound, upper_bound, int(outlier_count)


def create_histogram(df: pd.DataFrame, column: str) -> None:
    """Create and save a histogram."""

    plt.figure(figsize=(8, 5))
    plt.hist(df[column].dropna(), bins=30)
    plt.xlabel(column.capitalize())
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {column.capitalize()}")
    plt.tight_layout()

    output_file = PLOTS_DIR / f"{column}_histogram.png"
    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"Saved: {output_file}")


def create_boxplot(df: pd.DataFrame, column: str) -> None:
    """Create and save a boxplot."""

    plt.figure(figsize=(8, 5))
    plt.boxplot(df[column].dropna(), vert=False)
    plt.xlabel(column.capitalize())
    plt.title(f"Boxplot of {column.capitalize()}")
    plt.tight_layout()

    output_file = PLOTS_DIR / f"{column}_boxplot.png"
    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"Saved: {output_file}")


def analyze_fare(df: pd.DataFrame) -> None:
    """Calculate fare statistics and explain positive skew."""

    fare = df["fare"]

    mean_fare = fare.mean()
    median_fare = fare.median()
    mode_fare = fare.mode()

    print("\nFare statistics:")
    print(f"Mean: {mean_fare:.4f}")
    print(f"Median: {median_fare:.4f}")

    if not mode_fare.empty:
        print(f"Mode: {mode_fare.iloc[0]:.4f}")
    else:
        print("Mode: No mode found")

    print("\nInterpretation of fare skew:")
    print(
        "Fare is positively (right) skewed because most passengers "
        "paid relatively lower fares, while a smaller number of "
        "passengers paid very high fares. These high values create "
        "a longer tail on the right side of the distribution and "
        "pull the mean upward."
    )

    if mean_fare > median_fare:
        print(
            "The mean is greater than the median, which supports "
            "the presence of positive skew."
        )


def analyze_iqr(df: pd.DataFrame, column: str) -> None:
    """Calculate and display IQR outlier information."""

    iqr, lower_bound, upper_bound, outlier_count = calculate_iqr_outliers(
        df[column]
    )

    print(f"\n{column.upper()} IQR analysis:")
    print(f"Q1: {df[column].quantile(0.25):.4f}")
    print(f"Q3: {df[column].quantile(0.75):.4f}")
    print(f"IQR: {iqr:.4f}")
    print(f"Lower bound: {lower_bound:.4f}")
    print(f"Upper bound: {upper_bound:.4f}")
    print(f"Outlier count: {outlier_count}")


def main() -> None:
    """Run the complete univariate analysis."""

    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"{CSV_FILE} was not found. "
            "Run titanic_data.py and preprocess.py first."
        )

    # Use the cleaned local CSV.
    df = pd.read_csv(CSV_FILE)

    print("Titanic cleaned dataset loaded.")
    print(f"Shape: {df.shape}")

    # ---------------------------------------------------------
    # Age analysis
    # ---------------------------------------------------------
    print("\nCreating AGE plots...")
    create_histogram(df, "age")
    create_boxplot(df, "age")
    analyze_iqr(df, "age")

    # ---------------------------------------------------------
    # Fare analysis
    # ---------------------------------------------------------
    print("\nCreating FARE plots...")
    create_histogram(df, "fare")
    create_boxplot(df, "fare")
    analyze_iqr(df, "fare")

    # ---------------------------------------------------------
    # Fare statistics
    # ---------------------------------------------------------
    analyze_fare(df)

    print("\nUNIVARIATE ANALYSIS COMPLETE.")


if __name__ == "__main__":
    main()