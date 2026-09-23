"""
Module 1 - Data Pipeline

Step 6: Required SQL queries and pandas verification.
"""

import sqlite3
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
DATABASE_FILE = DATA_DIR / "books.db"
RESULTS_FILE = DATA_DIR / "sql_query_results.txt"


def run_query(connection, query, query_name, output_file):
    """Run a SQL query, display it, and persist its statement and rows."""

    print()
    print("=" * 60)
    print(query_name)
    print("=" * 60)

    result = pd.read_sql(query, connection)

    print(result.to_string(index=False))

    with output_file.open("a", encoding="utf-8") as file:
        file.write(f"{query_name}\n")
        file.write("=" * 60 + "\n")
        file.write("SQL statement:\n")
        file.write(query.strip() + "\n\n")
        file.write("Result rows:\n")
        file.write(result.to_string(index=False) + "\n\n")

    return result


def main():
    """Run all required SQL queries."""

    connection = sqlite3.connect(DATABASE_FILE)
    RESULTS_FILE.write_text(
        "SQL query results\n" + "=" * 60 + "\n\n",
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Query 1: SELECT + WHERE
    # Find books costing more than £40.
    # ---------------------------------------------------------

    query_1 = """
    SELECT title, price_gbp, rating
    FROM books
    WHERE price_gbp > 40
    ORDER BY price_gbp DESC
    """

    result_1 = run_query(
        connection,
        query_1,
        "QUERY 1 - SELECT + WHERE",
        RESULTS_FILE,
    )

    # ---------------------------------------------------------
    # Query 2: ORDER BY
    # Find the most expensive books.
    # ---------------------------------------------------------

    query_2 = """
    SELECT title, price_gbp, price_inr
    FROM books
    ORDER BY price_gbp DESC
    LIMIT 10
    """

    result_2 = run_query(
        connection,
        query_2,
        "QUERY 2 - ORDER BY + LIMIT",
        RESULTS_FILE,
    )

    # ---------------------------------------------------------
    # Query 3: DISTINCT
    # Show unique ratings.
    # ---------------------------------------------------------

    query_3 = """
    SELECT DISTINCT rating
    FROM books
    ORDER BY rating
    """

    result_3 = run_query(
        connection,
        query_3,
        "QUERY 3 - DISTINCT",
        RESULTS_FILE,
    )

    # ---------------------------------------------------------
    # Query 4: BETWEEN
    # Find books priced between £20 and £30.
    # ---------------------------------------------------------

    query_4 = """
    SELECT title, price_gbp, rating
    FROM books
    WHERE price_gbp BETWEEN 20 AND 30
    ORDER BY price_gbp
    """

    result_4 = run_query(
        connection,
        query_4,
        "QUERY 4 - BETWEEN",
        RESULTS_FILE,
    )

    # ---------------------------------------------------------
    # Query 5: IN
    # Find books from Mystery and Poetry.
    # ---------------------------------------------------------

    query_5 = """
    SELECT title, rating, price_gbp
    FROM books
    WHERE category_id IN (
        SELECT category_id
        FROM categories
        WHERE category_name IN ('Mystery', 'Poetry')
    )
    ORDER BY rating DESC, title
    """

    result_5 = run_query(
        connection,
        query_5,
        "QUERY 5 - IN",
        RESULTS_FILE,
    )

    # ---------------------------------------------------------
    # Query 6: JOIN
    # Combine books with their category names.
    # ---------------------------------------------------------

    query_6 = """
    SELECT
        books.book_id,
        books.title,
        books.price_gbp,
        books.rating,
        books.in_stock,
        categories.category_name
    FROM books
    JOIN categories
        ON books.category_id = categories.category_id
    ORDER BY books.book_id
    """

    result_6 = run_query(
        connection,
        query_6,
        "QUERY 6 - JOIN",
        RESULTS_FILE,
    )

    # ---------------------------------------------------------
    # Verification: SQL JOIN vs pandas merge
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("JOIN VERIFICATION - SQL vs PANDAS MERGE")
    print("=" * 60)

    # Read the two database tables into pandas.
    books_df = pd.read_sql(
        "SELECT * FROM books ORDER BY book_id",
        connection
    )

    categories_df = pd.read_sql(
        "SELECT * FROM categories ORDER BY category_id",
        connection
    )

    # Reproduce the SQL JOIN using pandas merge().
    merged_df = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner"
    )

    # Select the same columns used in Query 6.
    pandas_join = merged_df[
        [
            "book_id",
            "title",
            "price_gbp",
            "rating",
            "in_stock",
            "category_name",
        ]
    ].copy()

    pandas_join = pandas_join.sort_values(
        "book_id"
    ).reset_index(drop=True)

    sql_join = result_6.copy()

    sql_join = sql_join.sort_values(
        "book_id"
    ).reset_index(drop=True)

    # SQLite stores boolean values as 0/1.
    # Pandas may read the same values as integers.
    sql_join["in_stock"] = sql_join["in_stock"].astype(int)

    pandas_join["in_stock"] = pandas_join["in_stock"].astype(int)

    # Make sure column order and data types are comparable.
    sql_join = sql_join[
        [
            "book_id",
            "title",
            "price_gbp",
            "rating",
            "in_stock",
            "category_name",
        ]
    ]

    pandas_join = pandas_join[
        [
            "book_id",
            "title",
            "price_gbp",
            "rating",
            "in_stock",
            "category_name",
        ]
    ]

    # Compare the results.
    joins_match = sql_join.equals(pandas_join)

    print()
    print("SQL JOIN rows:", len(sql_join))
    print("Pandas merge rows:", len(pandas_join))
    print("SQL JOIN and pandas merge match:", joins_match)

    if joins_match:
        print("VERIFICATION PASSED")
    else:
        print("VERIFICATION FAILED")

    # ---------------------------------------------------------
    # Final verification
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("FINAL DATABASE CHECK")
    print("=" * 60)

    total_books = pd.read_sql(
        "SELECT COUNT(*) AS total_books FROM books",
        connection
    )

    total_categories = pd.read_sql(
        "SELECT COUNT(*) AS total_categories FROM categories",
        connection
    )

    print(
        "Total books:",
        total_books.loc[0, "total_books"]
    )

    print(
        "Total categories:",
        total_categories.loc[0, "total_categories"]
    )

    connection.close()


if __name__ == "__main__":
    main()
