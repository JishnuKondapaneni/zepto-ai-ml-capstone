# Module 2 — Titanic Analytics and Machine Learning

## 1. Dataset Loading and Export

The Titanic dataset was loaded using `sns.load_dataset("titanic")`.

The dataset was immediately exported to:

`analytics/titanic.csv`

All downstream analysis uses the local CSV file.

Original dataset:

- Rows: 891
- Columns: 15

Cleaned dataset:

- Rows: 889
- Columns: 14

---

## 2. Missing-Value Treatment

Missing values were handled according to the required percentage thresholds.

| Column | Missing Values | Treatment |
|---|---:|---|
| age | 177 | Median imputation |
| embarked | 2 | Rows dropped |
| deck | 688 | Column dropped |
| embark_town | 2 | Rows dropped |

### Explanation

`age` had approximately 19.87% missing values, which is within the 5–30% range. Therefore, missing ages were replaced using the median age of 28.0.

`embarked` and `embark_town` each had approximately 0.22% missing values, which is below 5%, so the affected rows were removed.

`deck` had approximately 77.22% missing values. Since more than 30% of its values were missing, the column was dropped. This avoids unreliable large-scale imputation.

After preprocessing, the cleaned dataset contains no missing values.

---

# 3. Univariate Analysis

## Age Distribution

The age histogram shows that passenger ages are concentrated mainly around young-adult and middle-aged groups.

The age boxplot identifies:

- Q1 = 22.0
- Q3 = 35.0
- IQR = 13.0
- Lower bound = 2.5
- Upper bound = 54.5
- IQR outliers = 65

The outliers above the upper IQR boundary represent passengers considerably older than the central age range.

## Fare Distribution

The fare histogram is strongly right-skewed. Most passengers paid relatively low fares, while a smaller number paid substantially higher fares.

The fare boxplot identifies:

- Q1 = 7.8958
- Q3 = 31.0
- IQR = 23.1042
- Lower bound = -26.7605
- Upper bound = 65.6563
- IQR outliers = 114

The negative lower bound is only the calculated IQR boundary; actual fares are not negative.

### Fare Mean, Median and Mode

- Mean = 32.0967
- Median = 14.4542
- Mode = 8.0500

The mean is considerably greater than the median because the fare distribution is positively skewed. A relatively small number of very expensive tickets creates a long right tail and pulls the mean upward.

---

# 4. Bivariate Analysis

## Survival by Sex

| Sex | Survival Rate |
|---|---:|
| Female | 74.04% |
| Male | 18.89% |

Female passengers had a substantially higher survival rate than male passengers.

This indicates that sex was strongly associated with survival in the Titanic dataset.

## Survival by Passenger Class

| Passenger Class | Survival Rate |
|---|---:|
| 1 | 62.62% |
| 2 | 47.28% |
| 3 | 24.24% |

First-class passengers had the highest survival rate, while third-class passengers had the lowest.

This suggests passenger class was an important factor associated with survival.

## Survival by Sex and Passenger Class

| Sex | Class | Survival Rate |
|---|---:|---:|
| Female | 1 | 96.74% |
| Female | 2 | 92.11% |
| Female | 3 | 50.00% |
| Male | 1 | 36.89% |
| Male | 2 | 15.74% |
| Male | 3 | 13.54% |

The combination of sex and passenger class provides more detail than either variable alone.

Female first- and second-class passengers had particularly high survival rates, while male second- and third-class passengers had substantially lower survival rates.

---

# 5. Correlation Analysis

The required correlation matrix uses exactly these six variables:

- survived
- pclass
- age
- sibsp
- parch
- fare

The two strongest absolute off-diagonal correlations are:

### 1. Passenger Class and Fare

Correlation:

`-0.5482`

This negative relationship occurs because passenger class is numerically encoded with smaller numbers representing higher classes. Higher-class passengers generally paid higher fares.

### 2. SibSp and Parch

Correlation:

`+0.4145`

This positive relationship indicates that passengers traveling with siblings/spouses were somewhat more likely to also travel with parents/children.

Correlation does not establish causation; it only describes the strength and direction of a linear relationship.

---

# 6. Multivariate Analysis

Four distinct multivariate visualizations were created:

1. Survival heatmap
2. Age vs. fare with survival information
3. Passenger class vs. fare with survival information
4. Family-size survival analysis

### Interpretation

The multivariate visualizations show that survival was influenced by combinations of passenger characteristics rather than by a single variable.

Sex and passenger class together reveal substantially different survival patterns. Fare also varies strongly across passenger classes, while family-related variables provide additional information about passenger circumstances.

The combined visualizations therefore provide more detailed insight than examining each variable independently.

---

# 7. Z-Score Standardization

Age and fare were standardized using z-score standardization.

### Before Standardization

| Variable | Mean | Standard Deviation |
|---|---:|---:|
| Age | 29.3152 | 12.9849 |
| Fare | 32.0967 | 49.6975 |

### After Standardization

| Variable | Mean | Standard Deviation |
|---|---:|---:|
| Age | approximately 0 | approximately 1 |
| Fare | approximately 0 | approximately 1 |

The verification used population standard deviation (`ddof=0`) and confirmed that both standardized variables have mean approximately 0 and standard deviation 1.

Standardization puts variables with different numerical scales onto a comparable scale. This is particularly useful for models such as logistic regression.

---

# 8. Classification Modeling

The target variable is:

`survived`

The data was split using a stratified 80/20 train-test split so that the survival-class proportions were preserved.

Preprocessing was implemented using a leakage-free `ColumnTransformer` and `Pipeline`.

Numeric features were processed using:

- Median imputation
- StandardScaler

Categorical features were processed using:

- Most-frequent imputation
- One-hot encoding

The `alive` column was removed because it directly represents survival and would cause target leakage.

## Classification Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8146 | 0.7966 | 0.6912 | 0.7402 | 0.8680 |
| Logistic Regression Balanced | **0.8315** | 0.7879 | **0.7647** | **0.7761** | 0.8674 |
| Logistic Regression + SMOTE | 0.8258 | 0.7937 | 0.7353 | 0.7634 | 0.8678 |
| Decision Tree | 0.7921 | 0.8163 | 0.5882 | 0.6838 | 0.8248 |
| Random Forest | 0.7865 | 0.7344 | 0.6912 | 0.7121 | 0.8146 |

### Interpretation

The balanced Logistic Regression model produced the strongest held-out test performance among the evaluated classifiers.

It achieved:

- Accuracy = 0.8315
- Precision = 0.7879
- Recall = 0.7647
- F1 = 0.7761
- ROC-AUC = 0.8674

Using `class_weight="balanced"` improved recall and F1 compared with the unbalanced Logistic Regression model.

SMOTE was also tested, with oversampling performed only within the training pipeline to prevent data leakage.

---

# 9. Random Forest Grid Search

GridSearchCV was used to tune the Random Forest.

The parameters searched were:

- `n_estimators`
- `max_depth`
- `max_features`

The best parameters were:

```text
n_estimators = 200
max_depth = 10
max_features = sqrt


Best cross-validation F1 score:

0.7593

The optimized Random Forest achieved on the test set:

Accuracy = 0.7921
Precision = 0.7719
Recall = 0.6471
F1 = 0.7040
ROC-AUC = 0.8128
OOB score = 0.8214

The GridSearchCV result is based on cross-validation F1, while the final comparison uses the independent test set. Therefore, the tuned Random Forest did not outperform the balanced Logistic Regression on the held-out test data.

10. Fare Regression

A multivariate Linear Regression model was created to predict passenger fare.

The target variable was:

fare

The target was excluded from the input features, and alive was also removed to avoid leakage.

The model used both numerical and categorical predictors with preprocessing contained inside a pipeline.

Regression Results
Metric	Result
MAE	18.3735
RMSE	41.2921
R²	0.3609
Adjusted R²	0.2655

The R² value of approximately 0.36 means that the model explains a meaningful but limited portion of the variation in passenger fares.

The difference between R² and adjusted R² reflects the adjustment for the number of predictors in the model.

A residual plot was also created to inspect heteroscedasticity. The spread of residuals is not completely constant across predicted fare values, indicating that the linear regression assumptions are not perfectly satisfied.

11. Saved Regression Pipeline

The complete preprocessing and regression model were saved using joblib.

Saved file:

analytics/fare_regression_pipeline.joblib

The saved pipeline was loaded back successfully and tested using raw feature data.

The loaded pipeline successfully produced a fare prediction:

Predicted fare: 2.0292

This confirms that the saved artifact contains the preprocessing and model components required to transform raw input and generate a prediction.

12. Generated Analysis Files

The analytics module generates the following important files:

analytics/
├── titanic.csv
├── titanic_cleaned.csv
├── classification_results.csv
├── random_forest_gridsearch_results.csv
├── regression_results.csv
├── fare_regression_pipeline.joblib
└── plots/
    ├── age_histogram.png
    ├── age_boxplot.png
    ├── fare_histogram.png
    ├── fare_boxplot.png
    ├── survival_by_sex.png
    ├── survival_by_class.png
    ├── survival_by_sex_and_class.png
    ├── correlation_matrix.png
    ├── multivariate_survival_heatmap.png
    ├── multivariate_age_fare_survival.png
    ├── multivariate_class_fare_survival.png
    ├── multivariate_family_survival.png
    ├── decision_tree.png
    ├── confusion_matrix_logistic_baseline.png
    ├── confusion_matrix_logistic_balanced.png
    ├── confusion_matrix_logistic_smote.png
    ├── confusion_matrix_decision_tree.png
    ├── confusion_matrix_random_forest.png
    ├── confusion_matrix_random_forest_gridsearch.png
    └── fare_residuals_heteroscedasticity.png
Module 2 Conclusion

The Titanic analysis demonstrates a complete analytics workflow:

Dataset acquisition and local export
Missing-value treatment
Univariate analysis
Bivariate analysis
Correlation analysis
Multivariate visualization
Feature standardization
Leakage-free classification modeling
Imbalance handling
Random Forest hyperparameter tuning
Fare regression
Residual analysis
Joblib model persistence and verification

The balanced Logistic Regression model provided the strongest held-out classification performance among the tested classifiers, while the Linear Regression model provided a baseline multivariate approach for predicting fare.