"""
Module 2 - Titanic Classification Modeling

Requirements:
- Stratified train/test split on survived.
- Leakage-free preprocessing fitted only on training data.
- Logistic Regression.
- Decision Tree with plot_tree.
- Random Forest.
- Confusion matrix.
- Accuracy, precision, recall, F1, ROC-AUC.
- Compare:
    1. Normal baseline
    2. class_weight='balanced'
    3. SMOTE applied only to the training data.
- Random Forest GridSearchCV.
- Tune:
    - n_estimators
    - max_depth
    - max_features
- Random Forest must use oob_score=True.
"""

from pathlib import Path

import matplotlib

# Use a non-GUI backend.
# This prevents Tkinter cleanup errors on Windows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.pipeline import Pipeline


# ============================================================
# FILE PATHS
# ============================================================

ANALYTICS_DIR = Path(__file__).resolve().parent

CSV_FILE = ANALYTICS_DIR / "titanic_cleaned.csv"

PLOTS_DIR = ANALYTICS_DIR / "plots"

PLOTS_DIR.mkdir(
    exist_ok=True
)

TARGET = "survived"


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor(
    X: pd.DataFrame,
) -> ColumnTransformer:
    """
    Create the preprocessing transformer.

    Numeric columns:
        - median imputation
        - standardization

    Categorical columns:
        - most-frequent imputation
        - one-hot encoding

    IMPORTANT:
    The transformer is fitted only inside the training
    pipeline, so test data is never used to fit preprocessing.
    """

    numeric_features = [
        "age",
        "sibsp",
        "parch",
        "fare",
    ]

    categorical_features = [
        "sex",
        "embarked",
        "class",
        "who",
        "adult_male",
        "embark_town",
        "alone",
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    return preprocessor


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model_name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate a fitted classification model.

    Metrics:
        - Accuracy
        - Precision
        - Recall
        - F1
        - ROC-AUC

    Also creates and saves a confusion matrix.
    """

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
    )

    recall = recall_score(
        y_test,
        predictions,
    )

    f1 = f1_score(
        y_test,
        predictions,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    print(
        f"\n{'=' * 60}"
    )

    print(model_name)

    print(
        f"{'=' * 60}"
    )

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )

    print(
        f"ROC-AUC:   {roc_auc:.4f}"
    )

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Did not survive",
                "Survived",
            ],
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    print(
        "Confusion matrix:"
    )

    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Did not survive",
            "Survived",
        ],
    )

    display.plot()

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.tight_layout()

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("=", "")
        .replace("'", "")
        .replace("-", "_")
    )

    output_file = (
        PLOTS_DIR
        / f"confusion_matrix_{safe_name}.png"
    )

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        "all"
    )

    print(
        f"Saved: {output_file}"
    )

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


# ============================================================
# DECISION TREE PLOT
# ============================================================

def create_decision_tree_plot(
    model: Pipeline,
    feature_names: list[str],
) -> None:
    """
    Plot the trained Decision Tree.

    Only the first 3 levels are displayed so that the
    visualization remains readable.
    """

    classifier = (
        model.named_steps["classifier"]
    )

    plt.figure(
        figsize=(20, 10)
    )

    plot_tree(
        classifier,
        feature_names=feature_names,
        class_names=[
            "Did not survive",
            "Survived",
        ],
        filled=False,
        max_depth=3,
        fontsize=8,
    )

    plt.title(
        "Decision Tree (First 3 Levels)"
    )

    plt.tight_layout()

    output_file = (
        PLOTS_DIR
        / "decision_tree.png"
    )

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        "all"
    )

    print(
        f"\nSaved: {output_file}"
    )


# ============================================================
# GET TRANSFORMED FEATURE NAMES
# ============================================================

def get_transformed_feature_names(
    model: Pipeline,
) -> list[str]:
    """
    Get feature names after ColumnTransformer preprocessing.
    """

    preprocessor = (
        model.named_steps["preprocessor"]
    )

    return list(
        preprocessor.get_feature_names_out()
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Run the complete classification analysis.
    """

    # ========================================================
    # CHECK INPUT FILE
    # ========================================================

    if not CSV_FILE.exists():

        raise FileNotFoundError(
            f"{CSV_FILE} was not found. "
            "Run titanic_data.py and preprocess.py first."
        )

    # ========================================================
    # LOAD CLEANED LOCAL CSV
    # ========================================================

    df = pd.read_csv(
        CSV_FILE
    )

    print(
        "Titanic cleaned dataset loaded."
    )

    print(
        f"Shape: {df.shape}"
    )

    # ========================================================
    # REMOVE DIRECT TARGET LEAKAGE
    # ========================================================

    columns_to_drop = [
        "alive",
    ]

    df = df.drop(
        columns=columns_to_drop
    )

    X = df.drop(
        columns=[TARGET]
    )

    y = df[TARGET]

    print(
        f"\nFeatures: {X.shape[1]}"
    )

    print(
        "Target distribution:"
    )

    print(
        y.value_counts()
    )

    print(
        "\nTarget proportions:"
    )

    print(
        y.value_counts(
            normalize=True
        ).round(4)
    )

    # ========================================================
    # STRATIFIED TRAIN / TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )

    print(
        "\nTRAIN/TEST SPLIT"
    )

    print(
        f"Training rows: {len(X_train)}"
    )

    print(
        f"Testing rows:  {len(X_test)}"
    )

    print(
        "\nTraining target distribution:"
    )

    print(
        y_train.value_counts(
            normalize=True
        ).round(4)
    )

    print(
        "\nTesting target distribution:"
    )

    print(
        y_test.value_counts(
            normalize=True
        ).round(4)
    )

    results = []

    # ========================================================
    # 1. LOGISTIC REGRESSION - BASELINE
    # ========================================================

    logistic_baseline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(
                    X_train
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    logistic_baseline.fit(
        X_train,
        y_train,
    )

    results.append(
        evaluate_model(
            "Logistic Regression - Baseline",
            logistic_baseline,
            X_test,
            y_test,
        )
    )

    # ========================================================
    # 2. LOGISTIC REGRESSION - BALANCED
    # ========================================================

    logistic_balanced = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(
                    X_train
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    logistic_balanced.fit(
        X_train,
        y_train,
    )

    results.append(
        evaluate_model(
            "Logistic Regression - Balanced",
            logistic_balanced,
            X_test,
            y_test,
        )
    )

    # ========================================================
    # 3. LOGISTIC REGRESSION - SMOTE
    # ========================================================

    logistic_smote = ImbPipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(
                    X_train
                ),
            ),
            (
                "smote",
                SMOTE(
                    random_state=42
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    logistic_smote.fit(
        X_train,
        y_train,
    )

    results.append(
        evaluate_model(
            "Logistic Regression - SMOTE",
            logistic_smote,
            X_test,
            y_test,
        )
    )

    # ========================================================
    # 4. DECISION TREE
    # ========================================================

    decision_tree = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(
                    X_train
                ),
            ),
            (
                "classifier",
                DecisionTreeClassifier(
                    random_state=42,
                    max_depth=5,
                ),
            ),
        ]
    )

    decision_tree.fit(
        X_train,
        y_train,
    )

    results.append(
        evaluate_model(
            "Decision Tree",
            decision_tree,
            X_test,
            y_test,
        )
    )

    feature_names = (
        get_transformed_feature_names(
            decision_tree
        )
    )

    create_decision_tree_plot(
        decision_tree,
        feature_names,
    )

    # ========================================================
    # 5. RANDOM FOREST
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "Random Forest"
    )

    print(
        "=" * 60
    )

    random_forest = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(
                    X_train
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    oob_score=True,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    random_forest.fit(
        X_train,
        y_train,
    )

    results.append(
        evaluate_model(
            "Random Forest",
            random_forest,
            X_test,
            y_test,
        )
    )

    rf_classifier = (
        random_forest.named_steps[
            "classifier"
        ]
    )

    print(
        f"\nRandom Forest OOB Score: "
        f"{rf_classifier.oob_score_:.4f}"
    )

    # ========================================================
    # 6. RANDOM FOREST GRIDSEARCHCV
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "Random Forest GridSearchCV"
    )

    print(
        "=" * 60
    )

    print(
        "\nStarting grid search..."
    )

    print(
        "This may take a little while."
    )

    # Pipeline ensures preprocessing is performed separately
    # inside each training fold.
    rf_grid_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(
                    X_train
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=42,
                    oob_score=True,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # Required parameters:
    # n_estimators
    # max_depth
    # max_features
    param_grid = {
        "classifier__n_estimators": [
            100,
            200,
        ],
        "classifier__max_depth": [
            None,
            5,
            10,
        ],
        "classifier__max_features": [
            "sqrt",
            "log2",
        ],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    grid_search = GridSearchCV(
        estimator=rf_grid_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        return_train_score=False,
    )

    grid_search.fit(
        X_train,
        y_train,
    )

    print(
        "\nGrid search complete."
    )

    print(
        "\nBest parameters:"
    )

    print(
        grid_search.best_params_
    )

    print(
        f"\nBest cross-validation F1: "
        f"{grid_search.best_score_:.4f}"
    )

    # ========================================================
    # BEST RANDOM FOREST
    # ========================================================

    best_random_forest = (
        grid_search.best_estimator_
    )

    results.append(
        evaluate_model(
            "Random Forest - GridSearchCV Best",
            best_random_forest,
            X_test,
            y_test,
        )
    )

    best_rf_classifier = (
        best_random_forest.named_steps[
            "classifier"
        ]
    )

    print(
        f"\nBest Random Forest OOB Score: "
        f"{best_rf_classifier.oob_score_:.4f}"
    )

    # ========================================================
    # SAVE GRID SEARCH RESULTS
    # ========================================================

    grid_results = pd.DataFrame(
        grid_search.cv_results_
    )

    grid_results_file = (
        ANALYTICS_DIR
        / "random_forest_gridsearch_results.csv"
    )

    grid_results.to_csv(
        grid_results_file,
        index=False,
    )

    print(
        f"\nSaved GridSearchCV results to: "
        f"{grid_results_file}"
    )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "MODEL COMPARISON"
    )

    print(
        "=" * 60
    )

    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # ========================================================
    # SAVE MODEL COMPARISON
    # ========================================================

    results_file = (
        ANALYTICS_DIR
        / "classification_results.csv"
    )

    results_df.to_csv(
        results_file,
        index=False,
    )

    print(
        f"\nSaved model comparison to: "
        f"{results_file}"
    )

    print(
        "\nCLASSIFICATION ANALYSIS COMPLETE."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()