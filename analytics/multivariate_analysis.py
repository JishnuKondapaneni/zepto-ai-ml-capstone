"""
Module 2 - Titanic Multivariate Analysis

Requirements:
- At least 4 distinct multivariate charts.
- Written interpretations for the charts.
- Before/after z-score standardization of age and fare.
- Standardized age and fare should have mean approximately 0
  and standard deviation approximately 1.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler


ANALYTICS_DIR = Path(__file__).resolve().parent
CSV_FILE = ANALYTICS_DIR / "titanic_cleaned.csv"
PLOTS_DIR = ANALYTICS_DIR / "plots"

PLOTS_DIR.mkdir(exist_ok=True)


def create_survival_heatmap(df: pd.DataFrame) -> None:
    """Show survival rate across sex and passenger class."""

    survival_table = pd.pivot_table(
        df,
        values="survived",
        index="sex",
        columns="pclass",
        aggfunc="mean",
    )

    plt.figure(figsize=(8, 5))

    sns.heatmap(
        survival_table * 100,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
    )

    plt.xlabel("Passenger Class")
    plt.ylabel("Sex")
    plt.title("Survival Rate (%) by Sex and Passenger Class")
    plt.tight_layout()

    output_file = PLOTS_DIR / "multivariate_survival_heatmap.png"
    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"Saved: {output_file}")


def create_age_fare_survival_plot(df: pd.DataFrame) -> None:
    """Show the relationship among age, fare, sex, and survival."""

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=df,
        x="age",
        y="fare",
        hue="survived",
        style="sex",
        alpha=0.7,
    )

    plt.xlabel("Age")
    plt.ylabel("Fare")
    plt.title("Age vs Fare by Survival and Sex")
    plt.tight_layout()

    output_file = PLOTS_DIR / "multivariate_age_fare_survival.png"
    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"Saved: {output_file}")


def create_class_fare_survival_plot(df: pd.DataFrame) -> None:
    """Show fare distributions by class and survival."""

    plt.figure(figsize=(9, 6))

    sns.boxplot(
        data=df,
        x="pclass",
        y="fare",
        hue="survived",
    )

    plt.xlabel("Passenger Class")
    plt.ylabel("Fare")
    plt.title("Fare Distribution by Passenger Class and Survival")
    plt.tight_layout()

    output_file = PLOTS_DIR / "multivariate_class_fare_survival.png"
    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"Saved: {output_file}")


def create_family_survival_plot(df: pd.DataFrame) -> None:
    """Show survival according to family-related variables."""

    family_df = df.copy()

    family_df["family_size"] = (
        family_df["sibsp"] + family_df["parch"] + 1
    )

    survival_by_family_size = (
        family_df.groupby("family_size")["survived"]
        .mean()
        .reset_index()
    )

    survival_by_family_size["survival_rate_percent"] = (
        survival_by_family_size["survived"] * 100
    )

    plt.figure(figsize=(9, 6))

    sns.lineplot(
        data=survival_by_family_size,
        x="family_size",
        y="survival_rate_percent",
        marker="o",
    )

    plt.xlabel("Family Size")
    plt.ylabel("Survival Rate (%)")
    plt.title("Survival Rate by Family Size")
    plt.tight_layout()

    output_file = PLOTS_DIR / "multivariate_family_survival.png"
    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"Saved: {output_file}")


def standardize_age_and_fare(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardize age and fare using z-score standardization.

    The scaler is fitted on the current cleaned dataset for this
    EDA demonstration. Modeling preprocessing will later be fitted
    only on the training data to prevent leakage.
    """

    columns = ["age", "fare"]

    before = df[columns].copy()

    scaler = StandardScaler()

    standardized_values = scaler.fit_transform(df[columns])

    after = pd.DataFrame(
        standardized_values,
        columns=["age_zscore", "fare_zscore"],
        index=df.index,
    )

    return before, after


def print_standardization_results(
    before: pd.DataFrame,
    after: pd.DataFrame,
) -> None:
    """Print before/after mean and standard deviation."""

    print("\nBEFORE STANDARDIZATION")
    print(
        before.agg(["mean", "std"]).round(4)
    )

    print("\nAFTER Z-SCORE STANDARDIZATION")
    print(
        after.agg(["mean", "std"]).round(4)
    )

    print("\nStandardization interpretation:")
    print(
        "Z-score standardization transforms each variable so that "
        "its mean is approximately 0 and its standard deviation is "
        "approximately 1. This puts age and fare on a comparable "
        "scale without changing the relative ordering of values."
    )


def main() -> None:
    """Run the complete multivariate analysis."""

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
    # Multivariate chart 1
    # ---------------------------------------------------------
    print("\nCreating multivariate survival heatmap...")
    create_survival_heatmap(df)

    # ---------------------------------------------------------
    # Multivariate chart 2
    # ---------------------------------------------------------
    print("\nCreating age/fare/survival/sex plot...")
    create_age_fare_survival_plot(df)

    # ---------------------------------------------------------
    # Multivariate chart 3
    # ---------------------------------------------------------
    print("\nCreating class/fare/survival plot...")
    create_class_fare_survival_plot(df)

    # ---------------------------------------------------------
    # Multivariate chart 4
    # ---------------------------------------------------------
    print("\nCreating family-size/survival plot...")
    create_family_survival_plot(df)

    # ---------------------------------------------------------
    # Z-score standardization
    # ---------------------------------------------------------
    print("\nRunning z-score standardization...")

    before, after = standardize_age_and_fare(df)

    print_standardization_results(before, after)

    # Verify approximately mean 0 and standard deviation 1.
    # pandas uses sample standard deviation (ddof=1), while
    # StandardScaler uses population standard deviation (ddof=0).
    # Therefore we verify using ddof=0.
    population_means = after.mean()
    population_stds = after.std(ddof=0)

    assert all(abs(value) < 1e-10 for value in population_means)

    assert all(
        abs(value - 1) < 1e-10
        for value in population_stds
    )

    print("\nStandardization verification PASSED.")

    print("\nMULTIVARIATE ANALYSIS COMPLETE.")


if __name__ == "__main__":
    main()