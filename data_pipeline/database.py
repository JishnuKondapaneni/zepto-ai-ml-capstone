"""
Module 1 - Data Pipeline

Step 5: Create normalized SQLite database
from the cleaned books CSV.
"""

import sqlite3

import pandas as pd


CSV_FILE = "data_pipeline/cleaned_books.csv"
DATABASE_FILE = "data_pipeline/books.db"


def load_cleaned_data():
    """Load the cleaned books CSV into a pandas DataFrame."""

    df = pd.read_csv(CSV_FILE)

    print("=" * 60)
    print("LOADED CLEANED DATA")
    print("=" * 60)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    return df


def validate_data(df):
    """Validate the cleaned dataset before inserting into SQLite."""

    print()
    print("=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)

    missing_values = df.isnull().sum()

    print("Missing values:")
    print(missing_values)

    # The scraper already produced a complete dataset.
    if missing_values.sum() == 0:
        print()
        print("No missing values found.")
        print("No median imputation is required.")
    else:
        print()
        print("Missing values were found.")

        # Median imputation for numeric columns.
        numeric_columns = [
            "price_gbp",
            "price_inr",
            "rating",
        ]

        for column in numeric_columns:
            if df[column].isnull().any():
                median_value = df[column].median()
                df[column] = df[column].fillna(median_value)

                print(
                    f"Median imputation applied to {column}: "
                    f"{median_value}"
                )

    return df


def create_database(df):
    """Create the required two-table normalized SQLite database."""

    connection = sqlite3.connect(DATABASE_FILE)

    cursor = connection.cursor()

    # Enable foreign-key enforcement.
    cursor.execute("PRAGMA foreign_keys = ON")

    # Remove existing tables so the script can be safely re-run.
    cursor.execute("DROP TABLE IF EXISTS books")
    cursor.execute("DROP TABLE IF EXISTS categories")

    # ---------------------------------------------------------
    # Table 1: categories
    # ---------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY,
            category_name TEXT UNIQUE
        )
        """
    )

    # ---------------------------------------------------------
    # Table 2: books
    # ---------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY,
            title TEXT,
            price_gbp REAL,
            price_inr REAL,
            rating INTEGER,
            in_stock INTEGER,
            category_id INTEGER REFERENCES categories(category_id)
        )
        """
    )

    # ---------------------------------------------------------
    # Insert categories
    # ---------------------------------------------------------

    categories = sorted(
        df["category"].dropna().unique()
    )

    for category in categories:

        cursor.execute(
            """
            INSERT INTO categories (category_name)
            VALUES (?)
            """,
            (category,)
        )

    # ---------------------------------------------------------
    # Insert books
    # ---------------------------------------------------------

    for _, row in df.iterrows():

        cursor.execute(
            """
            SELECT category_id
            FROM categories
            WHERE category_name = ?
            """,
            (row["category"],)
        )

        category_result = cursor.fetchone()

        if category_result is None:
            raise ValueError(
                f"Category not found: {row['category']}"
            )

        category_id = category_result[0]

        cursor.execute(
            """
            INSERT INTO books (
                title,
                price_gbp,
                price_inr,
                rating,
                in_stock,
                category_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row["title"],
                row["price_gbp"],
                row["price_inr"],
                row["rating"],
                int(row["in_stock"]),
                category_id,
            )
        )

    connection.commit()

    print()
    print("=" * 60)
    print("DATABASE CREATED")
    print("=" * 60)

    print("Database:", DATABASE_FILE)

    # Verify number of rows in both tables.
    cursor.execute("SELECT COUNT(*) FROM categories")
    category_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM books")
    book_count = cursor.fetchone()[0]

    print("Categories:", category_count)
    print("Books:", book_count)

    connection.close()


def main():
    """Run the database creation pipeline."""

    df = load_cleaned_data()

    df = validate_data(df)

    create_database(df)


if __name__ == "__main__":
    main()