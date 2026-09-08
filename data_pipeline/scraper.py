"""
Module 1 - Data Pipeline

Step 4: Scrape books from multiple categories and pages.
"""

import csv
import re

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://books.toscrape.com/"

# Three categories required by the assignment.
CATEGORY_URLS = {
    "Travel": "https://books.toscrape.com/catalogue/category/books/travel_2/index.html",
    "Mystery": "https://books.toscrape.com/catalogue/category/books/mystery_3/index.html",
    "Poetry": "https://books.toscrape.com/catalogue/category/books/poetry_23/index.html",
}


RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


# Fixed exchange rate required by the assignment.
GBP_TO_INR = 105.50


def get_page(url):
    """Download a webpage and return BeautifulSoup."""
    response = requests.get(url, timeout=15)
    response.raise_for_status()

    # Force UTF-8 so characters such as curly apostrophes are decoded correctly.
    response.encoding = "utf-8"

    return BeautifulSoup(response.text, "html.parser")


def extract_book(book, category):
    """Extract required information from one book."""

    title_element = book.select_one("h3 a")
    price_element = book.select_one(".price_color")
    rating_element = book.select_one("p.star-rating")
    availability_element = book.select_one(".availability")

    # Skip a book if the title is missing.
    if not title_element:
        return None

    title = title_element.get("title", "").strip()

    price = ""
    if price_element:
        price = price_element.get_text(strip=True)

    rating = ""
    if rating_element:
        classes = rating_element.get("class", [])
        if len(classes) >= 2:
            rating = classes[1]

    availability = ""
    if availability_element:
        availability = availability_element.get_text(
            " ",
            strip=True
        )

    relative_url = title_element.get("href", "")

    product_url = requests.compat.urljoin(
        BASE_URL,
        relative_url
    )

    return {
        "title": title,
        "category": category,
        "price_gbp": price,
        "rating": rating,
        "availability": availability,
        "product_url": product_url,
    }


def scrape_category(category, category_url):
    """Scrape all available pages for one category."""

    category_books = []
    page_url = category_url
    page_number = 1

    while page_url:

        print(
            f"  Scraping {category} - page {page_number}"
        )

        soup = get_page(page_url)

        books = soup.select("article.product_pod")

        print(
            f"  Books found: {len(books)}"
        )

        for book in books:

            book_data = extract_book(
                book,
                category
            )

            if book_data:
                category_books.append(book_data)

        # Check whether another page exists.
        next_button = soup.select_one("li.next a")

        if next_button:

            next_relative_url = next_button.get(
                "href",
                ""
            )

            page_url = requests.compat.urljoin(
                page_url,
                next_relative_url
            )

            page_number += 1

        else:
            page_url = None

    return category_books


def scrape_all_categories():
    """Scrape all configured categories."""

    all_books = []

    for category, category_url in CATEGORY_URLS.items():

        print()
        print("=" * 60)
        print(f"Category: {category}")
        print("=" * 60)

        category_books = scrape_category(
            category,
            category_url
        )

        print(
            f"Total books in {category}: "
            f"{len(category_books)}"
        )

        all_books.extend(category_books)

    return all_books


def clean_price(price):
    """Convert a price such as '£45.17' into a float."""

    if not price:
        return None

    cleaned = (
        price
        .replace("Â", "")
        .replace("£", "")
        .strip()
    )

    match = re.search(
        r"\d+(?:\.\d+)?",
        cleaned
    )

    if match:
        return float(match.group())

    return None


def clean_rating(rating):
    """Convert text rating into a numeric value from 1 to 5."""

    return RATING_MAP.get(rating)


def clean_in_stock(availability):
    """
    Convert availability text into a boolean.

    'In stock' -> True
    'In stock (22 available)' -> True
    Missing or unavailable -> False
    """

    if not availability:
        return False

    if "In stock" in availability:
        return True

    return False


def clean_book(book):
    """Clean one scraped book and convert GBP price to INR."""

    cleaned_book = book.copy()

    # Clean GBP price.
    cleaned_book["price_gbp"] = clean_price(
        book["price_gbp"]
    )

    # Convert rating text to number.
    cleaned_book["rating"] = clean_rating(
        book["rating"]
    )

    # Convert availability to boolean.
    cleaned_book["in_stock"] = clean_in_stock(
        book["availability"]
    )

    # Remove the raw availability field.
    del cleaned_book["availability"]

    # Convert GBP to INR using the fixed assignment rate.
    if cleaned_book["price_gbp"] is not None:

        cleaned_book["price_inr"] = round(
            cleaned_book["price_gbp"] * GBP_TO_INR,
            2
        )

    else:

        cleaned_book["price_inr"] = None

    return cleaned_book


def clean_books(books):
    """Clean all books and remove duplicate product URLs."""

    cleaned_books = []
    seen_urls = set()

    for book in books:

        cleaned_book = clean_book(book)

        product_url = cleaned_book["product_url"]

        # Skip records without a product URL.
        if not product_url:
            continue

        # Remove duplicate books using product URL.
        if product_url in seen_urls:
            continue

        seen_urls.add(product_url)

        cleaned_books.append(cleaned_book)

    return cleaned_books


def save_cleaned_books(cleaned_books):
    """Save cleaned books to a CSV file."""

    output_file = "data_pipeline/cleaned_books.csv"

    fieldnames = [
        "title",
        "category",
        "price_gbp",
        "price_inr",
        "rating",
        "in_stock",
        "product_url",
    ]

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(cleaned_books)

    print()
    print(
        f"Cleaned dataset saved to: {output_file}"
    )


def main():
    """Run the scraping, cleaning, and saving process."""

    # Step 1: Scrape books.
    books = scrape_all_categories()

    print()
    print("=" * 60)
    print("RAW SCRAPING SUMMARY")
    print("=" * 60)

    print(
        "Raw books scraped:",
        len(books)
    )

    # Step 2: Clean books.
    cleaned_books = clean_books(books)

    print()
    print("=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)

    print(
        "Clean books:",
        len(cleaned_books)
    )

    # Display first cleaned book.
    print()
    print("First cleaned book:")

    print(
        cleaned_books[0]
    )

    # Display last cleaned book.
    print()
    print("Last cleaned book:")

    print(
        cleaned_books[-1]
    )

    # Step 3: Save cleaned dataset.
    save_cleaned_books(cleaned_books)


if __name__ == "__main__":
    main()