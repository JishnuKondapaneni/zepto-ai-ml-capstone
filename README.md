# Zepto AI/ML Capstone

This repository contains three independent modules completed for the Zepto AI/ML capstone assignment:

1. `data_pipeline` - web scraping, cleaning, SQLite database design, and SQL/Pandas verification.
2. `analytics` - Titanic exploratory analysis, classification, regression, model evaluation, and saved ML pipelines.
3. `support_assistant` - retrieval-augmented customer support assistant using local embeddings, ChromaDB, LangGraph, FastAPI, and an optional OpenRouter LLM path.

All modules can be run locally with Python. No paid services are required.

## Repository Structure

```text
zepto-ai-ml-capstone/
|
+-- data_pipeline/
|   +-- scraper.py
|   +-- database.py
|   +-- verify_database.py
|   +-- sql_queries.py
|   +-- cleaned_books.csv
|   +-- books.db
|   +-- README.md
|
+-- analytics/
|   +-- titanic_data.py
|   +-- preprocess.py
|   +-- univariate_analysis.py
|   +-- bivariate_analysis.py
|   +-- multivariate_analysis.py
|   +-- classification.py
|   +-- regression.py
|   +-- save_regression_model.py
|   +-- save_best_classification_model.py
|   +-- titanic.csv
|   +-- titanic_cleaned.csv
|   +-- classification_results.csv
|   +-- random_forest_gridsearch_results.csv
|   +-- regression_results.csv
|   +-- fare_regression_pipeline.joblib
|   +-- best_classification_pipeline.joblib
|   +-- plots/
|   +-- README.md
|
+-- support_assistant/
|   +-- app.py
|   +-- graph.py
|   +-- ingest.py
|   +-- models.py
|   +-- prompt_template.py
|   +-- retriever.py
|   +-- Dockerfile
|   +-- docs/
|   +-- chroma_db/
|   +-- README.md
|
+-- requirements.txt
+-- README.md
```

The `.venv` virtual environment is local-only and is not part of the repository.

---

# Setup

## 1. Clone the repository

```bash
git clone https://github.com/JishnuKondapaneni/zepto-ai-ml-capstone.git
cd zepto-ai-ml-capstone
```

## 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```powershell
pip install -r requirements.txt
```

The project uses Python 3.12.

---

# Module 1 - Data Pipeline

## Objective

The data pipeline scrapes book information from Books to Scrape, cleans the collected data, stores it in a normalized SQLite database, and verifies SQL results against Pandas results.

## Source

Books to Scrape:

```text
https://books.toscrape.com/
```

The scraper collects books from three categories:

* Travel
* Mystery
* Poetry

The pipeline collected 62 books in total.

## Required fields

The raw data includes:

* title
* raw GBP price
* star rating text
* availability text
* category

The cleaned dataset includes:

* title
* category
* price_gbp
* price_inr
* rating
* in_stock
* product_url

## Cleaning

The pipeline:

* converts prices to numeric GBP values
* converts star ratings to integers from 1 to 5
* converts availability to a boolean `in_stock`
* converts GBP to INR using the fixed rate:

```text
1 GBP = 105.50 INR
```

* removes duplicate records
* handles missing numeric values using median imputation where required

## SQLite Database

The database contains exactly two normalized tables:

* `categories`
* `books`

The `books.category_id` column references the primary key of `categories`.

## SQL verification

Six SQL queries are included and cover:

* SELECT and WHERE
* ORDER BY
* LIMIT
* DISTINCT
* IN and BETWEEN
* JOIN

The JOIN result is also reproduced using `pd.merge`.

The SQL JOIN and Pandas merge both return 62 rows and are verified as equivalent.

## Run Module 1

From the repository root:

```powershell
python data_pipeline/scraper.py
python data_pipeline/database.py
python data_pipeline/verify_database.py
python data_pipeline/sql_queries.py
```

More details are available in:

```text
data_pipeline/README.md
```

---

# Module 2 - Analytics

## Objective

The analytics module performs exploratory data analysis and machine learning using the Titanic dataset.

## Dataset loading

The Titanic dataset is loaded using Seaborn:

```python
sns.load_dataset("titanic")
```

The dataset is immediately saved to:

```text
analytics/titanic.csv
```

Downstream analysis reads the saved CSV rather than repeatedly downloading the dataset.

The dataset contains 891 rows and 15 original columns.

## Missing-value treatment

Missing values were handled according to the assignment thresholds:

* less than 5% missing: drop affected rows
* 5% to 30% missing: impute
* more than 30% missing: drop the column with justification

For this dataset:

* `age` had approximately 19.87% missing values and was median-imputed.
* `embarked` had approximately 0.22% missing values and affected rows were dropped.
* `embark_town` had approximately 0.22% missing values and affected rows were dropped.
* `deck` had approximately 77.22% missing values and was dropped because the missingness exceeded 30%.

The cleaned dataset is saved as:

```text
analytics/titanic_cleaned.csv
```

## Univariate analysis

The analysis includes:

* Age histogram
* Age box plot
* Fare histogram
* Fare box plot
* IQR outlier counts for age and fare
* Fare mean, median, and mode
* Fare skewness interpretation

Detected IQR outliers:

* Age: 65
* Fare: 114

Fare has a positive/right-skewed distribution because the mean is substantially higher than the median.

## Bivariate analysis

The analysis includes survival comparisons by:

* sex
* passenger class
* sex and passenger class

The required 6 x 6 correlation matrix uses exactly:

```text
survived
pclass
age
sibsp
parch
fare
```

The two strongest absolute off-diagonal correlations are:

1. `pclass` and `fare`: approximately -0.5482
2. `sibsp` and `parch`: approximately +0.4145

## Multivariate analysis

The module includes at least four distinct multivariate visualizations with written interpretations.

Age and fare were standardized using z-scores.

Before standardization:

```text
Age mean: 29.3152
Age standard deviation: 12.9849

Fare mean: 32.0967
Fare standard deviation: 49.6975
```

After standardization, both variables have means approximately equal to zero and population standard deviations approximately equal to one.

## Classification

The classification workflow uses leakage-free preprocessing inside Scikit-learn pipelines.

A stratified train/test split with `random_state=42` is used.

Models evaluated include:

* Logistic Regression
* balanced Logistic Regression
* Logistic Regression with SMOTE
* Decision Tree
* Random Forest
* tuned Random Forest using GridSearchCV

The evaluation includes:

* accuracy
* precision
* recall
* F1 score
* ROC curve
* AUC
* confusion matrix

The imbalance experiments include:

* baseline classification
* `class_weight="balanced"`
* SMOTE applied only to the training data

### Best held-out classifier

The balanced Logistic Regression model produced the best held-out classification performance among the evaluated models.

Its test results were:

```text
Accuracy:  0.8315
Precision: 0.7879
Recall:    0.7647
F1:        0.7761
AUC:       0.8674
```

The final saved classification pipeline is:

```text
analytics/best_classification_pipeline.joblib
```

It contains both preprocessing and the estimator and can be reloaded to make predictions from raw input columns.

## Random Forest GridSearchCV

The Random Forest grid search evaluates:

```text
n_estimators: [100, 200]
max_depth: [None, 5, 10]
max_features: [sqrt, log2]
```

The estimator uses:

```text
oob_score=True
```

The best grid-search parameters were:

```text
max_depth: 10
max_features: sqrt
n_estimators: 200
```

The best tuned Random Forest achieved an OOB score of approximately:

```text
0.8214
```

Grid-search results are saved in:

```text
analytics/random_forest_gridsearch_results.csv
```

## Regression

A multivariate linear regression model predicts `fare`.

Reported metrics include:

```text
MAE: 18.3735
RMSE: 41.2921
R2: 0.3609
Adjusted R2: 0.2655
```

Residual analysis is used to assess heteroscedasticity.

The complete regression pipeline is saved as:

```text
analytics/fare_regression_pipeline.joblib
```

## Run Module 2

From the repository root:

```powershell
python analytics/titanic_data.py
python analytics/preprocess.py
python analytics/univariate_analysis.py
python analytics/bivariate_analysis.py
python analytics/multivariate_analysis.py
python analytics/classification.py
python analytics/regression.py
python analytics/save_regression_model.py
python analytics/save_best_classification_model.py
```

The analysis outputs and charts are stored under:

```text
analytics/
analytics/plots/
```

More details are available in:

```text
analytics/README.md
```

---

# Module 3 - Support Assistant

## Objective

The support assistant is a retrieval-augmented customer support application focused on Zepto policy questions.

It uses:

* local Sentence Transformers embeddings
* ChromaDB
* LangGraph
* FastAPI
* Pydantic
* an optional OpenRouter LLM path

No paid API is required for the graded baseline.

## Knowledge base

The policy documents are stored in:

```text
support_assistant/docs/
```

The documents are ingested into ChromaDB using:

```text
support_assistant/ingest.py
```

The local embedding model is:

```text
all-MiniLM-L6-v2
```

## LangGraph workflow

The graph contains three main nodes:

```text
classify_intent
retrieve_and_answer
direct_answer
```

The routing logic classifies questions using the required policy keywords.

Policy-question keywords include:

```text
delivery
return
refund
membership
tracking
cancel
gift card
support hours
```

Policy questions are routed through retrieval.

General questions use the fixed baseline response:

```text
I can only answer questions about Zepto policies right now.
```

## Retrieval

For policy questions, the assistant retrieves the top 3 relevant chunks using cosine similarity.

In the graded mock mode, the answer follows the required deterministic format:

```text
Based on the retrieved context: {top_chunk_snippet}
```

The response also contains source document/chunk IDs and a deterministic confidence score.

## Structured response

FastAPI returns a Pydantic `QueryResponse` containing:

```text
answer
sources
confidence
```

## Mock LLM mode

The graded baseline uses:

```text
MOCK_LLM=1
```

or the default behavior when `MOCK_LLM` is unset.

This makes the baseline deterministic and avoids dependence on paid external services.

## Optional real LLM mode

An optional OpenRouter path is available when:

```text
MOCK_LLM=0
```

An API key is required for the real LLM path.

Do not commit API keys or other secrets to GitHub.

## FastAPI

The application exposes:

```text
POST /ask
```

Example request:

```json
{
  "query": "How long does delivery take?"
}
```

Example policy response structure:

```json
{
  "answer": "Based on the retrieved context: ...",
  "sources": [
    "document/chunk"
  ],
  "confidence": 1.0
}
```

## Run Module 3 locally

From the repository root:

```powershell
python support_assistant/ingest.py
python support_assistant/app.py
```

The FastAPI application runs on port 7860.

Swagger documentation is available at:

```text
http://127.0.0.1:7860/docs
```

## Docker

Build the image:

```powershell
docker build -t zepto-support-assistant ./support_assistant
```

Run the container:

```powershell
docker run --rm -p 7860:7860 zepto-support-assistant
```

The Docker image uses mock mode by default:

```text
MOCK_LLM=1
```

This allows the application to run without an external LLM API key.

---

# End-to-End Run Order

From a clean checkout, run the modules in this order.

## Module 1

```powershell
python data_pipeline/scraper.py
python data_pipeline/database.py
python data_pipeline/verify_database.py
python data_pipeline/sql_queries.py
```

## Module 2

```powershell
python analytics/titanic_data.py
python analytics/preprocess.py
python analytics/univariate_analysis.py
python analytics/bivariate_analysis.py
python analytics/multivariate_analysis.py
python analytics/classification.py
python analytics/regression.py
python analytics/save_regression_model.py
python analytics/save_best_classification_model.py
```

## Module 3

```powershell
python support_assistant/ingest.py
python support_assistant/app.py
```

Then open:

```text
http://127.0.0.1:7860/docs
```

---

# Design Decisions

## Data Pipeline

Books to Scrape was selected because it provides a stable public HTML dataset suitable for demonstrating requests and BeautifulSoup scraping.

The database uses separate category and book tables to avoid repeating category names and to demonstrate a normalized relational design with a foreign key.

A fixed GBP-to-INR conversion rate of 105.50 was used exactly as required by the assignment.

## Analytics

The Titanic CSV is created immediately after the initial Seaborn load so downstream steps can work from a reproducible local artifact.

Preprocessing is placed inside Scikit-learn pipelines for model training so that transformations are fitted only on training data and data leakage is avoided.

The balanced Logistic Regression model is selected as the best held-out classifier because it achieved the strongest test-set F1 score and also improved recall compared with the unbalanced baseline.

## Support Assistant

Local embeddings and ChromaDB provide a no-cost retrieval layer.

LangGraph makes the routing between policy retrieval and general-question handling explicit.

Mock LLM mode is the default graded path because it is deterministic and does not require a paid external API.

The optional real LLM path can be enabled separately without changing the retrieval architecture.

---

# Git Workflow

The project was developed using a feature branch and merged back into `main`.

The repository includes:

* multiple commits
* a feature branch
* a non-fast-forward merge commit
* the completed work pushed to `main`

The merge history can be inspected with:

```powershell
git log --graph --oneline --decorate --all
```

The final repository is intended to be submitted as the public GitHub repository:

```text
https://github.com/JishnuKondapaneni/zepto-ai-ml-capstone
```

---

# Final Deliverables

The repository contains all three required capstone modules:

```text
/data_pipeline
/analytics
/support_assistant
README.md
requirements.txt
```

Each module contains its implementation, supporting outputs, and module-specific README documentation.

The project is designed to be reproducible locally without paid services.
