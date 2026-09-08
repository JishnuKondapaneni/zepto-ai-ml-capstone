import sqlite3


DATABASE_FILE = "data_pipeline/books.db"


connection = sqlite3.connect(DATABASE_FILE)


print("=" * 60)
print("DATABASE VERIFICATION")
print("=" * 60)


# Check tables
print()
print("Tables:")

tables = connection.execute(
    """
    SELECT name
    FROM sqlite_master
    WHERE type = 'table'
    ORDER BY name
    """
).fetchall()

print(tables)


# Check categories
print()
print("Categories:")

categories = connection.execute(
    """
    SELECT *
    FROM categories
    ORDER BY category_id
    """
).fetchall()

for category in categories:
    print(category)


# Check first 5 books
print()
print("First 5 books:")

books = connection.execute(
    """
    SELECT *
    FROM books
    ORDER BY book_id
    LIMIT 5
    """
).fetchall()

for book in books:
    print(book)


# Check total books
print()
print("Total books:")

book_count = connection.execute(
    """
    SELECT COUNT(*)
    FROM books
    """
).fetchone()[0]

print(book_count)


connection.close()