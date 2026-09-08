"""
Module 2 - Titanic Bivariate Analysis

Requirements:
- Survival by sex.
- Survival by passenger class.
- Survival by sex and passenger class.
- Exact 6x6 correlation matrix using:
    survived, pclass, age, sibsp, parch, fare
- Identify and explain the top 2 absolute off-diagonal correlations.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ANALYTICS_DIR = Path(__file__).resolve().parent
CSV_FILE = ANALYTICS_DIR / "titanic_cleaned.csv"
PLOTS_DIR = ANALYTICS_DIR / "plots"

PLOTS_DIR.mkdir(exist_ok=True)


CORRELATION_COLUMNS = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
]


def survival_by_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate survival rate by sex."""

    result = df.groupby("sex")["survived"].mean().reset_index()

    result["survival_rate_percent"] = result["survived"] * 100

    return result


def survival_by_class(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate survival rate by passenger class."""

    result = df.groupby("pclass")["survived"].mean().reset_index()

    result["survival_rate_percent"] = result["survived"] * 100

    return result


def survival_by_sex_and_class(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate survival rate by sex and passenger class."""

    result = (
        df.groupby(["sex", "pclass"])["survived"]
        .mean()
        .reset_index()
    )

    result["survival_rate_percent"] = result["survived"] * 100

    return result


def create_survival_by_sex_plot(df: pd.DataFrame) -> None:
    """Create survival-rate plot by sex."""

    result = survival_by_sex(df)

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=result,
        x="sex",
        y="survival_rate_percent",
    )

    plt.xlabel("Sex")
    plt.ylabel("Survival Rate (%)")
    plt.title("Survival Rate by Sex")
    plt.ylim(0, 100)
    plt.tight_layout()

    output_file = PLOTS_DIR / "survival_by_sex.png"
    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"Saved: {output_file}")


def create_survival_by_class_plot(df: pd.DataFrame) -> None:
    """Create survival-rate plot by passenger class."""

    result = survival_by_class(df)

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=result,
        x="pclass",
        y="survival_rate_percent",
    )

    plt.xlabel("Passenger Class")
    plt.ylabel("Survival Rate (%)")
    plt.title("Survival Rate by Passenger Class")
    plt.ylim(0, 100)
    plt.tight_layout()

    output_file = PLOTS_DIR / "survival_by_class.png"
    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"Saved: {output_file}")


def create_survival_by_sex_and_class_plot(df: pd.DataFrame) -> None:
    """Create survival-rate plot by sex and passenger class."""

    result = survival_by_sex_and_class(df)

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=result,
        x="pclass",
        y="survival_rate_percent",
        hue="sex",
    )

    plt.xlabel("Passenger Class")
    plt.ylabel("Survival Rate (%)")
    plt.title("Survival Rate by Sex and Passenger Class")
    plt.ylim(0, 100)
    plt.tight_layout()

    output_file = PLOTS_DIR / "survival_by_sex_and_class.png"
    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"Saved: {output_file}")


def calculate_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the required exact 6x6 correlation matrix."""

    correlation_matrix = df[CORRELATION_COLUMNS].corr()

    return correlation_matrix


def find_top_two_correlations(
    correlation_matrix: pd.DataFrame,
) -> list[tuple[str, str, float]]:
    """
    Find the top two absolute off-diagonal correlations.

    Diagonal values of 1.0 are excluded.
    """

    correlations = []

    columns = correlation_matrix.columns

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            column_1 = columns[i]
            column_2 = columns[j]

            value = correlation_matrix.loc[column_1, column_2]

            correlations.append(
                (column_1, column_2, value)
            )

    correlations.sort(
        key=lambda item: abs(item[2]),
        reverse=True,
    )

    return correlations[:2]


def create_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
) -> None:
    """Create and save the required 6x6 correlation heatmap."""

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )

    plt.title("Titanic Correlation Matrix")
    plt.tight_layout()

    output_file = PLOTS_DIR / "correlation_matrix.png"
    plt.savefig(output_file, dpi=150)
    plt.close()

    print(f"Saved: {output_file}")


def print_interpretations(
    top_correlations: list[tuple[str, str, float]],
) -> None:
    """Print interpretations of the two strongest correlations."""

    print("\nTop 2 absolute off-diagonal correlations:")

    for rank, (column_1, column_2, value) in enumerate(
        top_correlations,
        start=1,
    ):
        print(
            f"{rank}. {column_1} vs {column_2}: "
            f"{value:.4f} "
            f"(absolute correlation = {abs(value):.4f})"
        )

    print("\nCorrelation interpretations:")

    for column_1, column_2, value in top_correlations:

        direction = "positive" if value > 0 else "negative"

        strength = abs(value)

        if strength >= 0.7:
            strength_text = "strong"
        elif strength >= 0.4:
            strength_text = "moderate"
        else:
            strength_text = "weak"

        print(
            f"- {column_1} and {column_2} have a "
            f"{strength_text} {direction} correlation "
            f"({value:.4f})."
        )

    print(
        "\nInterpretation note: correlation measures the strength "
        "and direction of a linear relationship between two numeric "
        "variables. Correlation does not prove causation."
    )


def main() -> None:
    """Run the complete bivariate analysis."""

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
    # Survival by sex
    # ---------------------------------------------------------
    print("\nSURVIVAL BY SEX")
    result_sex = survival_by_sex(df)
    print(result_sex.to_string(index=False))

    create_survival_by_sex_plot(df)

    # ---------------------------------------------------------
    # Survival by passenger class
    # ---------------------------------------------------------
    print("\nSURVIVAL BY PASSENGER CLASS")
    result_class = survival_by_class(df)
    print(result_class.to_string(index=False))

    create_survival_by_class_plot(df)

    # ---------------------------------------------------------
    # Survival by sex and passenger class
    # ---------------------------------------------------------
    print("\nSURVIVAL BY SEX AND PASSENGER CLASS")
    result_sex_class = survival_by_sex_and_class(df)
    print(result_sex_class.to_string(index=False))

    create_survival_by_sex_and_class_plot(df)

    # ---------------------------------------------------------
    # Exact 6x6 correlation matrix
    # ---------------------------------------------------------
    print("\n6x6 CORRELATION MATRIX")

    correlation_matrix = calculate_correlation_matrix(df)

    print(correlation_matrix)

    # Verify exact dimensions.
    assert correlation_matrix.shape == (6, 6)

    # Verify exact required columns.
    assert list(correlation_matrix.columns) == CORRELATION_COLUMNS

    create_correlation_heatmap(correlation_matrix)

    # ---------------------------------------------------------
    # Top 2 correlations
    # ---------------------------------------------------------
    top_correlations = find_top_two_correlations(
        correlation_matrix
    )

    print_interpretations(top_correlations)

    print("\nBIVARIATE ANALYSIS COMPLETE.")


if __name__ == "__main__":
    main()