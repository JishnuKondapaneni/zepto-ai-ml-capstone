# Analytics Module

This module performs exploratory data analysis, preprocessing, classification modeling, and regression modeling on the Titanic dataset.

## 1. Dataset Loading

The Titanic dataset is loaded using Seaborn exactly once in `titanic_data.py`:

```python
sns.load_dataset("titanic")
```

Immediately after loading, the dataset is saved as:

```text
analytics/titanic.csv
```

All downstream analysis reads from the saved CSV rather than repeatedly downloading/loading the dataset.

Dataset size:

* Rows: 891
* Columns: 15

---

## 2. Missing Value Analysis and Preprocessing

Missing values in the original Titanic dataset were handled according to the assignment thresholds.

| Column      | Missing | Percentage | Action            |
| ----------- | ------: | ---------: | ----------------- |
| age         |     177 |     19.87% | Median imputation |
| embarked    |       2 |      0.22% | Drop rows         |
| deck        |     688 |     77.22% | Drop column       |
| embark_town |       2 |      0.22% | Drop rows         |

The `age` column has between 5% and 30% missing values, so missing ages were replaced with the median value of 28.0.

The `embarked` and `embark_town` columns each have less than 5% missing values, so their affected rows were removed.

The `deck` column has more than 30% missing values. It was therefore removed because retaining it would require extensive imputation and could introduce substantial artificial information.

After preprocessing, the cleaned dataset contains 889 rows and 14 columns with no remaining missing values.

The cleaned dataset is saved as:

```text
analytics/titanic_cleaned.csv
```

---

## 3. Univariate Analysis

### Age

An age histogram and box plot were generated.

The IQR method identified 65 potential age outliers. These observations were retained because they are valid passenger ages rather than obvious data-entry errors.

### Fare

A fare histogram and box plot were generated.

The IQR method identified 114 potential fare outliers. These observations were also retained because unusually expensive tickets are plausible in the Titanic dataset.

Fare summary statistics:

| Statistic |    Fare |
| --------- | ------: |
| Mean      | 32.0967 |
| Median    | 14.4542 |
| Mode      |    8.05 |

The fare distribution is strongly right-skewed because most passengers paid relatively low fares while a smaller number paid substantially higher fares. This causes the mean to be considerably larger than the median.

Generated plots:

```text
analytics/age_histogram.png
analytics/age_boxplot.png
analytics/fare_histogram.png
analytics/fare_boxplot.png
```

---

## 4. Bivariate Analysis

Survival was analyzed by:

* Sex
* Passenger class
* Sex and passenger class together

The analysis shows substantial differences in survival between passenger groups. Female passengers generally had higher survival rates than male passengers, while passengers in higher classes generally had better survival outcomes.

A 6x6 Pearson correlation matrix was calculated using exactly these variables:

```text
survived
pclass
age
sibsp
parch
fare
```

The two strongest absolute off-diagonal correlations were:

| Variables       | Correlation |
| --------------- | ----------: |
| pclass and fare |     -0.5482 |
| sibsp and parch |      0.4145 |

The negative `pclass`-`fare` relationship occurs because lower numerical passenger-class values represent higher classes, which generally had more expensive fares.

The positive `sibsp`-`parch` relationship indicates that passengers traveling with siblings/spouses were also somewhat more likely to travel with parents/children.

Generated plots:

```text
analytics/survival_by_sex.png
analytics/survival_by_pclass.png
analytics/survival_by_sex_pclass.png
analytics/correlation_heatmap.png
```

---

## 5. Multivariate Analysis

Four distinct multivariate visualizations were produced.

### Chart 1: Survival by Sex and Passenger Class

This chart combines two important passenger characteristics and shows that survival varied substantially by both sex and passenger class. Female passengers generally had better survival outcomes than males across passenger classes. The combination demonstrates that survival cannot be explained by sex or class independently.

### Chart 2: Age and Fare by Survival

This visualization compares age and fare distributions for survivors and non-survivors. Fare shows noticeable differences between the groups, while age distributions overlap considerably. This suggests that fare-related passenger characteristics may contain more direct predictive information than age alone.

### Chart 3: Survival by Sex and Class

The grouped visualization makes the interaction between sex and passenger class more visible. Female passengers in higher classes had particularly strong survival outcomes, while male passengers generally had lower survival rates. The chart demonstrates why combining multiple categorical variables can provide more useful insight than examining either variable independently.

### Chart 4: Multivariate Correlation Heatmap

The heatmap summarizes relationships among survived, pclass, age, sibsp, parch, and fare simultaneously. The strongest relationship is the negative correlation between passenger class and fare, followed by the positive relationship between siblings/spouses and parents/children. Survival also has meaningful relationships with passenger class and fare, supporting their inclusion in classification models.

Generated multivariate plots:

```text
analytics/multivariate_survival_sex_class.png
analytics/multivariate_age_fare_survival.png
analytics/multivariate_survival_sex_pclass.png
analytics/multivariate_correlation.png
```

---

## 6. Z-Score Standardization

Age and fare were standardized using z-score standardization.

Before standardization:

| Variable |    Mean | Standard Deviation |
| -------- | ------: | -----------------: |
| age      | 29.3152 |            12.9849 |
| fare     | 32.0967 |            49.6975 |

After standardization, the population means were approximately zero and the population standard deviations were approximately one.

The pandas sample standard deviation is approximately 1.0006 because pandas uses `ddof=1`, while the standardization transformation uses the population standard deviation convention.

This confirms that the z-score transformation was applied correctly.

---

# 7. Classification Modeling

The classification target is:

```text
survived
```

The `alive` column was removed to prevent target leakage because it directly represents the survival outcome.

A stratified train/test split was used with:

```text
test_size = 0.20
random_state = 42
```

Preprocessing was implemented using a leakage-free `ColumnTransformer` and `Pipeline`.

Numerical features:

```text
age
sibsp
parch
fare
```

Numerical preprocessing:

```text
median imputation
StandardScaler
```

Categorical features:

```text
sex
embarked
class
who
adult_male
embark_town
alone
```

Categorical preprocessing:

```text
most-frequent imputation
OneHotEncoder
```

All preprocessing steps were fitted only on the training data through the pipeline.

---

## 8. Classification Results

| Model                             |   Accuracy | Precision |     Recall |         F1 | ROC-AUC |
| --------------------------------- | ---------: | --------: | ---------: | ---------: | ------: |
| Logistic Regression - Baseline    |     0.8146 |    0.7966 |     0.6912 |     0.7402 |  0.8680 |
| Logistic Regression - Balanced    | **0.8315** |    0.7879 | **0.7647** | **0.7761** |  0.8674 |
| Logistic Regression - SMOTE       |     0.8258 |    0.7937 |     0.7353 |     0.7634 |  0.8678 |
| Decision Tree                     |     0.7921 |    0.8163 |     0.5882 |     0.6838 |  0.8248 |
| Random Forest                     |     0.7865 |    0.7344 |     0.6912 |     0.7121 |  0.8146 |
| Random Forest - GridSearchCV Best |     0.7921 |    0.7719 |     0.6471 |     0.7040 |  0.8128 |

The balanced Logistic Regression model achieved the best held-out accuracy and F1 score among the evaluated classifiers.

---

## 9. Confusion Matrix and ROC Curves

Confusion matrices were generated for all evaluated classification models.

ROC curves were generated using the test-set probabilities and `roc_curve`, with ROC-AUC reported for each classifier.

Generated ROC plots:

```text
analytics/roc_curve_logistic_regression___baseline.png
analytics/roc_curve_logistic_regression___balanced.png
analytics/roc_curve_logistic_regression___smote.png
analytics/roc_curve_decision_tree.png
analytics/roc_curve_random_forest.png
analytics/roc_curve_random_forest___gridsearchcv_best.png
```

The ROC-AUC values show that the logistic regression models provided the strongest ranking performance, with the baseline Logistic Regression reaching an AUC of 0.8680.

---

## 10. Random Forest GridSearchCV

Random Forest hyperparameters were tuned using `GridSearchCV`.

Search parameters:

```text
n_estimators: [100, 200]
max_depth: [None, 5, 10]
max_features: [sqrt, log2]
```

The search used:

```text
5-fold StratifiedKFold
scoring = f1
```

The best parameters were:

```text
n_estimators = 200
max_depth = 10
max_features = sqrt
```

Best cross-validation F1:

```text
0.7593
```

The best GridSearchCV Random Forest achieved an independent test F1 score of 0.7040 and accuracy of 0.7921.

The final GridSearchCV Random Forest estimator used:

```python
oob_score=True
```

Its out-of-bag score was:

```text
0.8214
```

---

## 11. Imbalance Handling

Three approaches were compared:

1. Logistic Regression baseline
2. Logistic Regression with `class_weight="balanced"`
3. Logistic Regression with SMOTE applied only to the training fold

The balanced Logistic Regression model produced the strongest held-out F1 score of 0.7761 and recall of 0.7647.

SMOTE also improved recall compared with the baseline, but its F1 score of 0.7634 was lower than the balanced Logistic Regression result.

---

# 12. Regression Modeling

A multivariate Linear Regression model was used to predict:

```text
fare
```

The regression pipeline includes preprocessing and the estimator so that transformations are learned only from the training data.

Test-set metrics:

| Metric      |   Value |
| ----------- | ------: |
| MAE         | 18.3735 |
| RMSE        | 41.2921 |
| R²          |  0.3609 |
| Adjusted R² |  0.2655 |

The R² value of 0.3609 indicates that the model explains approximately 36.09% of the variation in fare on the test data.

The difference between R² and adjusted R² reflects the penalty applied for the number of predictors in the model.

The residual analysis indicates some heteroscedasticity, meaning that residual variability is not completely constant across predicted fare values.

Generated plot:

```text
analytics/fare_regression_residuals.png
```

---

# 13. Final Model Comparison

## Classification Metrics

| Model                             |   Accuracy | Precision |     Recall |         F1 | ROC-AUC |
| --------------------------------- | ---------: | --------: | ---------: | ---------: | ------: |
| Logistic Regression - Baseline    |     0.8146 |    0.7966 |     0.6912 |     0.7402 |  0.8680 |
| Logistic Regression - Balanced    | **0.8315** |    0.7879 | **0.7647** | **0.7761** |  0.8674 |
| Logistic Regression - SMOTE       |     0.8258 |    0.7937 |     0.7353 |     0.7634 |  0.8678 |
| Decision Tree                     |     0.7921 |    0.8163 |     0.5882 |     0.6838 |  0.8248 |
| Random Forest                     |     0.7865 |    0.7344 |     0.6912 |     0.7121 |  0.8146 |
| Random Forest - GridSearchCV Best |     0.7921 |    0.7719 |     0.6471 |     0.7040 |  0.8128 |

## Regression Metrics

| Model                          |     MAE |    RMSE |     R² | Adjusted R² |
| ------------------------------ | ------: | ------: | -----: | ----------: |
| Multivariate Linear Regression | 18.3735 | 41.2921 | 0.3609 |      0.2655 |

---

# 14. Final Recommendation

The recommended classification model is Logistic Regression with `class_weight="balanced"`, because it achieved the highest held-out accuracy of 0.8315 and the highest F1 score of 0.7761 among the evaluated classifiers. It also achieved a strong ROC-AUC of 0.8674 and recall of 0.7647, making it a good choice when correctly identifying survivors is important. The Random Forest GridSearchCV model achieved an OOB score of 0.8214, but its independent test F1 of 0.7040 and accuracy of 0.7921 were lower than the balanced Logistic Regression results. For fare prediction, the Linear Regression model achieved MAE = 18.3735, RMSE = 41.2921, and R² = 0.3609, indicating useful but limited predictive power and some residual heteroscedasticity.

---

# 15. Saved Model Artifacts

The best-performing classification pipeline is saved as:

analytics/best_classification_pipeline.joblib

This is the balanced Logistic Regression model selected from the evaluated
classification models.

The saved classification pipeline contains both preprocessing and the
Logistic Regression estimator, allowing raw input data to be passed directly
after reloading.

The pipeline was reloaded with joblib and verified using raw test-set input.
The verification successfully produced a binary prediction and survival
probability.

The fare regression pipeline is also saved as:

analytics/fare_regression_pipeline.joblib

The regression pipeline contains preprocessing and the regression estimator
and was similarly reloaded and verified with raw input.---

# 16. Main Analytics Files

```text
analytics/
├── README.md
├── titanic_data.py
├── preprocess.py
├── univariate_analysis.py
├── bivariate_analysis.py
├── multivariate_analysis.py
├── classification.py
├── regression.py
├── save_regression_model.py
├── titanic.csv
├── titanic_cleaned.csv
├── fare_regression_pipeline.joblib
└── generated PNG charts
```

---

# 17. How to Run

From the repository root, activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run the dataset loading step:

```powershell
python .\analytics\titanic_data.py
```

Run preprocessing:

```powershell
python .\analytics\preprocess.py
```

Run univariate analysis:

```powershell
python .\analytics\univariate_analysis.py
```

Run bivariate analysis:

```powershell
python .\analytics\bivariate_analysis.py
```

Run multivariate analysis:

```powershell
python .\analytics\multivariate_analysis.py
```

Run classification:

```powershell
python .\analytics\classification.py
```

Run regression:

```powershell
python .\analytics\regression.py
```

Save and verify the regression pipeline:

```powershell
python .\analytics\save_regression_model.py
```

---

# 18. Conclusion

The analytics module provides a complete workflow from dataset acquisition and cleaning through exploratory analysis, classification, regression, evaluation, and model persistence.

The analysis demonstrates that passenger sex and class are important survival-related variables, while fare has a substantial relationship with passenger class.

Among the evaluated classification models, balanced Logistic Regression provides the strongest held-out performance.

The fare regression model provides useful predictive information but explains only part of the observed fare variation, so its predictions should be interpreted with appropriate caution.
