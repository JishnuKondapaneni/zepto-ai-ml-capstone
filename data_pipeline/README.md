# Module 1 — Data Pipeline

## 1. Overview

This module implements a complete data pipeline for scraping book information from Books to Scrape, cleaning the scraped data, storing it in a normalized SQLite database, and verifying SQL results using pandas.

Source website:

`https://books.toscrape.com/`

The pipeline consists of:

1. Web scraping
2. Data cleaning
3. Currency conversion
4. Missing-value validation and imputation
5. CSV export
6. SQLite database creation
7. SQL querying
8. SQL-to-pandas verification

---

## 2. Web Scraping

The scraper collects book information from the Books to Scrape website.

The required fields extracted are:

- Title
- Price
- Star rating
- Availability
- Category

The scraper collected books from three categories:

- Travel
- Mystery
- Poetry

Total books scraped:

`62`

The scraped data is saved to:

`data_pipeline/cleaned_books.csv`

The scraper also stores the product URL in the CSV for traceability.

---

## 3. Data Cleaning

The raw scraped values were converted into analysis-ready formats.

### Price

The raw GBP price is cleaned by removing the pound symbol and converting the value to a floating-point number.

For example:

`£45.17`

becomes:

`45.17`

The cleaned field is:

`price_gbp`

### Rating

The website provides ratings as words such as:

- One
- Two
- Three
- Four
- Five

These were converted to numeric values from 1 to 5.

### Availability

The availability text was converted into a Boolean field:

- In stock → `True`
- Not in stock → `False`

The database stores this Boolean value as SQLite integer values:

- `1` = True
- `0` = False

### Price Conversion

A fixed exchange rate was used:

`1 GBP = 105.50 INR`

Therefore:

`price_inr = price_gbp × 105.50`

The exact exchange rate is fixed for reproducibility and is not retrieved dynamically.

---

## 4. Missing and Duplicate Data

The cleaned dataset was checked for missing values.

The final dataset contains no missing values in:

- title
- category
- price_gbp
- price_inr
- rating
- in_stock
- product_url

No median imputation was required because the scraped dataset contained no missing values after cleaning.

Duplicate product URLs were also checked.

Duplicate URLs:

`0`

Final cleaned dataset:

- Rows: 62
- Columns: 7

---

## 5. SQLite Database

The cleaned data was stored in:

`data_pipeline/books.db`

The database uses exactly two normalized tables:

### categories

```sql
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY,
    category_name TEXT UNIQUE
);


books
CREATE TABLE books (
    book_id INTEGER PRIMARY KEY,
    title TEXT,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock INTEGER,
    category_id INTEGER REFERENCES categories(category_id)
);

The category_id field connects each book to its category.

This separates category information from book information and avoids repeating category names for every book.

6. Database Verification

The SQLite database was successfully created and verified.

Database contents:

Categories: 3
Books: 62

The database contains the following categories:

Mystery
Poetry
Travel

Foreign-key support is enabled in SQLite.

The database was also checked to confirm that both required tables exist.

7. SQL Queries

Six SQL queries were implemented to demonstrate the required SQL operations.

Query 1 — SELECT and WHERE

Books with prices greater than £40 were selected.

This demonstrates filtering records using WHERE.

Query 2 — ORDER BY and LIMIT

The 10 most expensive books were selected.

This demonstrates:

ORDER BY
LIMIT
Query 3 — DISTINCT

Distinct book ratings were selected.

This demonstrates the DISTINCT keyword.

Query 4 — BETWEEN

Books with prices between £20 and £30 were selected.

This demonstrates the BETWEEN operator.

Query 5 — IN

Books belonging to the Mystery or Poetry categories were selected using:

IN

Query 6 — JOIN

Books were joined with the categories table using category_id.

This demonstrates a relational SQL JOIN.

8. SQL and Pandas Verification

SQL query results were loaded into pandas using:

pd.read_sql()

The SQL JOIN result was independently reproduced using:

pd.merge()

The verification produced:

SQL JOIN rows: 62
Pandas merge rows: 62

The SQL JOIN and pandas merge results matched.

Verification result:

True

Therefore, the SQL and pandas implementations produced equivalent JOIN results.

9. Files

The main files in this module are:

data_pipeline/
├── scraper.py
├── database.py
├── verify_database.py
├── sql_queries.py
├── cleaned_books.csv
├── books.db
└── README.md
scraper.py

Scrapes book information from Books to Scrape and performs data cleaning.

database.py

Creates the normalized SQLite database and inserts the cleaned book and category data.

verify_database.py

Checks the database tables, categories, sample records, and total book count.

sql_queries.py

Runs the required SQL queries and verifies the SQL JOIN against a pandas merge.

cleaned_books.csv

Contains the cleaned scraped book dataset.

books.db

Contains the normalized SQLite database.

10. Module 1 Conclusion

The data pipeline successfully performs the complete workflow from web scraping to structured database storage and verification.

The final pipeline produced:

62 scraped books
3 categories
Clean numeric prices
Numeric ratings from 1 to 5
Boolean stock status
INR price conversion using the fixed rate of 105.50 INR per GBP
A normalized two-table SQLite database
Six SQL queries
SQL JOIN verification using pandas

The module demonstrates web scraping, data cleaning, relational database design, SQL querying, and pandas-based verification.