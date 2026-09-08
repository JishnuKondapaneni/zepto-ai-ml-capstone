# Zepto AI/ML Capstone Project

This repository contains a three-module AI/ML capstone project covering data engineering, analytics and machine learning, and a policy-focused customer support assistant.

## Project Structure

```text
zepto-ai-ml-capstone/
│
├── data_pipeline/
│   ├── scraper.py
│   ├── database.py
│   ├── verify_database.py
│   ├── sql_queries.py
│   ├── cleaned_books.csv
│   ├── books.db
│   └── README.md
│
├── analytics/
│   ├── titanic_data.py
│   ├── preprocess.py
│   ├── univariate_analysis.py
│   ├── bivariate_analysis.py
│   ├── multivariate_analysis.py
│   ├── classification.py
│   ├── regression.py
│   ├── save_regression_model.py
│   ├── plots/
│   ├── titanic.csv
│   ├── titanic_cleaned.csv
│   ├── fare_regression_results.csv
│   ├── classification_results.csv
│   ├── regression_residuals.png
│   ├── fare_regression_pipeline.joblib
│   └── README.md
│
├── support_assistant/
│   ├── docs/
│   │   ├── doc_01.txt
│   │   ├── doc_02.txt
│   │   ├── doc_03.txt
│   │   ├── doc_04.txt
│   │   ├── doc_05.txt
│   │   ├── doc_06.txt
│   │   ├── doc_07.txt
│   │   └── doc_08.txt
│   ├── ingest.py
│   ├── retriever.py
│   ├── graph.py
│   ├── models.py
│   ├── prompt_template.py
│   ├── app.py
│   ├── Dockerfile
│   └── README.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

* Python 3.12 tested locally
* Git
* Docker Desktop is required only for the Docker deployment test
* No paid services are required

## Setup

Open PowerShell in the repository root.

Create the virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

All commands below should be run from the repository root unless otherwise stated.

---

# Module 1 — Data Pipeline

The data pipeline scrapes book information from `books.toscrape.com`, cleans the data, stores it in a normalized SQLite database, and verifies SQL and pandas operations.

## Run the scraper

```powershell
python data_pipeline/scraper.py
```

The scraper collects books from three categories and produces:

```text
data_pipeline/cleaned_books.csv
```

The cleaned dataset contains:

* title
* category
* price_gbp
* price_inr
* rating
* in_stock
* product_url

The INR conversion uses the fixed assignment rate:

```text
1 GBP = 105.50 INR
```

## Create the SQLite database

```powershell
python data_pipeline/database.py
```

This creates:

```text
data_pipeline/books.db
```

The database contains exactly two normalized tables:

* `categories`
* `books`

The `books.category_id` column references `categories.category_id`.

## Verify the database

```powershell
python data_pipeline/verify_database.py
```

## Run SQL and pandas verification

```powershell
python data_pipeline/sql_queries.py
```

This executes multiple SQL queries covering filtering, ordering, limiting, distinct values, range/category filtering, and joins.

It also compares the SQL JOIN result with an equivalent `pandas.merge()` result.

---

# Module 2 — Analytics and Machine Learning

The analytics module uses the Titanic dataset to perform data cleaning, exploratory data analysis, statistical analysis, classification, and regression.

## Run in this order

First load and save the original Titanic dataset:

```powershell
python analytics/titanic_data.py
```

Clean the dataset:

```powershell
python analytics/preprocess.py
```

Run univariate analysis:

```powershell
python analytics/univariate_analysis.py
```

Run bivariate analysis:

```powershell
python analytics/bivariate_analysis.py
```

Run multivariate analysis and standardization checks:

```powershell
python analytics/multivariate_analysis.py
```

Run classification models:

```powershell
python analytics/classification.py
```

Run fare regression:

```powershell
python analytics/regression.py
```

Save and verify the complete regression pipeline:

```powershell
python analytics/save_regression_model.py
```

## Main analytics outputs

The module produces:

* cleaned Titanic data
* histograms
* boxplots
* survival comparison charts
* correlation analysis
* multivariate visualizations
* classification metrics
* decision tree visualization
* ROC/confusion-matrix analysis
* regression metrics
* regression residual analysis
* a saved Joblib regression pipeline

### Classification

The classification workflow uses leakage-free preprocessing with scikit-learn pipelines.

The models include:

* Logistic Regression
* Decision Tree
* Random Forest

The workflow evaluates:

* accuracy
* precision
* recall
* F1 score
* ROC-AUC

Class imbalance is investigated using:

* baseline Logistic Regression
* class-weight-balanced Logistic Regression
* SMOTE applied only to the training data

Random Forest hyperparameters are tuned using `GridSearchCV`, and out-of-bag scoring is also reported.

### Regression

Fare is predicted using multivariate linear regression.

The evaluation includes:

* MAE
* RMSE
* R²
* Adjusted R²
* residual analysis

The complete preprocessing and regression workflow is saved as:

```text
analytics/fare_regression_pipeline.joblib
```

---

# Module 3 — Support Assistant

The support assistant is a local Retrieval-Augmented Generation (RAG) system for answering questions about Zepto policies.

## Architecture

```text
Policy Documents
      │
      ▼
Document Ingestion
      │
      ▼
Sentence Transformer
all-MiniLM-L6-v2
      │
      ▼
ChromaDB
      │
      ▼
User Question
      │
      ▼
LangGraph
      │
      ├── policy_question
      │        │
      │        ▼
      │   Top-3 Retrieval
      │        │
      │        ▼
      │   Answer Generation
      │
      └── general_question
               │
               ▼
        Policy-only response
```

## Policy documents

Exactly eight policy documents are stored in:

```text
support_assistant/docs/
```

They cover:

1. Delivery
2. Returns and refunds
3. Membership
4. Tracking
5. Cancellation
6. Damaged or missing items
7. Gift cards
8. Support hours

## Build the local vector database

Before running the assistant for the first time, ingest the policy documents:

```powershell
python -m support_assistant.ingest
```

This creates the local ChromaDB data under:

```text
support_assistant/chroma_db/
```

The Chroma database is a generated local artifact and is intentionally excluded from Git.

If the vector database is deleted or needs to be rebuilt, run the ingestion command again.

## Run the FastAPI application

The graded baseline uses the deterministic mock LLM mode:

```powershell
$env:MOCK_LLM="1"
```

Start the API:

```powershell
python -m uvicorn support_assistant.app:app --host 127.0.0.1 --port 7860
```

Open the interactive API documentation in a browser:

```text
http://127.0.0.1:7860/docs
```

The API provides:

```text
POST /ask
```

Example request:

```json
{
  "query": "How long does delivery take?"
}
```

Example policy response:

```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee.",
  "sources": [
    "doc_01",
    "doc_08",
    "doc_04"
  ],
  "confidence": 1.0
}
```

Example general-question response:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

## Mock LLM behavior

`MOCK_LLM` defaults to `1`.

When mock mode is enabled:

* policy questions perform real ChromaDB retrieval
* the top retrieved policy context is used to construct the deterministic answer
* confidence is `1.0`
* source document IDs are returned
* general questions receive the required policy-only response

This provides a deterministic, local, no-cost graded baseline.

## Optional real LLM mode

The code also contains an optional real LLM path.

Set:

```powershell
$env:MOCK_LLM="0"
```

The real path expects an OpenAI-compatible API configuration and validates the generated response using Pydantic.

The real LLM path retries up to two additional times when response generation or schema validation fails.

The graded baseline does not require a paid LLM service.

---

# Docker Deployment

The support assistant includes a Dockerfile based on Python 3.11.

## Build the image

Run from the repository root:

```powershell
docker build -f support_assistant/Dockerfile -t zepto-support-assistant .
```

## Run the container

```powershell
docker run --rm -p 7860:7860 zepto-support-assistant
```

The API is then available at:

```text
http://127.0.0.1:7860/docs
```

The Docker image uses:

```text
MOCK_LLM=1
```

by default.

The local embedding model is loaded when the application starts. The first startup can take longer because the model may need to be downloaded.

---

# Design Decisions

## Data Pipeline

A normalized two-table SQLite schema was used to avoid repeating category names for every book. Foreign-key relationships connect books to categories.

The scraper performs cleaning before database insertion. Numeric missing values are handled using median imputation when required.

## Analytics

Missing-value handling follows the assignment thresholds.

* Small amounts of missing data are handled by dropping affected rows.
* Moderate missingness is handled with median/mode imputation.
* Columns with very high missingness are removed when justified.

All model preprocessing is performed inside scikit-learn pipelines so transformations are fitted using training data only.

This prevents data leakage between training and testing datasets.

## Support Assistant

The support assistant uses:

* `all-MiniLM-L6-v2` for local embeddings
* ChromaDB for vector storage
* cosine similarity for retrieval
* LangGraph for workflow orchestration
* FastAPI for the API
* Pydantic for structured response validation

The mock mode is intentionally deterministic so the required baseline can be tested without paid external services.

---

# Verification

The three modules were tested locally.

Verified:

* Data scraping and cleaning
* SQLite database creation
* SQL queries and pandas JOIN equivalence
* Titanic dataset loading and preprocessing
* Univariate, bivariate, and multivariate analysis
* Classification models and evaluation metrics
* Fare regression and saved Joblib pipeline
* Local policy ingestion
* ChromaDB retrieval
* LangGraph routing
* FastAPI `/ask` endpoint
* Policy and general-question responses
* Docker image build
* Docker container startup

## Cost

This project does not require paid services.

The support assistant's graded baseline runs locally using:

```text
MOCK_LLM=1
```

and a locally loaded embedding model.

---

# Git Workflow

The project was developed using a feature branch:

```text
feature/capstone-complete
```

The feature branch contains multiple commits and is merged into `main`.

The repository is intended to be hosted publicly on GitHub.

---

# License

This project was created as an educational capstone assignment.
