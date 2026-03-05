"""
Titanic Survival Prediction - Enterprise Grade ML Pipeline

Author: Your Name
Description:
Reproducible, leakage-safe, modular ML pipeline using
ColumnTransformer, Pipeline, cross-validation, and artifact tracking.
"""

from __future__ import annotations

import os
import logging
import warnings
from dataclasses import dataclass
from typing import Tuple

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

warnings.filterwarnings("ignore")

# =====================================================
# Configuration
# =====================================================

@dataclass
class Config:
    train_path: str = "data/train.csv"
    test_path: str = "data/test.csv"
    results_dir: str = "results"
    model_path: str = "results/model.joblib"
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5


CFG = Config()
os.makedirs(CFG.results_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# =====================================================
# Data Utilities
# =====================================================

def load_data(path: str) -> pd.DataFrame:
    logging.info(f"Loading data: {path}")
    return pd.read_csv(path)


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop(columns=["Name", "Ticket", "Cabin"], errors="ignore", inplace=True)
    return df


# =====================================================
# Pipeline Builder
# =====================================================

def build_pipeline(random_state: int) -> Pipeline:

    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        n_jobs=None
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])


# =====================================================
# Evaluation & Visualization
# =====================================================

def evaluate_model(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> Tuple[float, float]:

    y_pred = pipeline.predict(X_val)
    y_prob = pipeline.predict_proba(X_val)[:, 1]

    accuracy = accuracy_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_prob)

    logging.info(f"Validation Accuracy: {accuracy:.4f}")
    logging.info(f"Validation ROC-AUC: {roc_auc:.4f}")
    logging.info("\n" + classification_report(y_val, y_pred))

    save_confusion_matrix(confusion_matrix(y_val, y_pred))
    save_roc_curve(y_val, y_prob)
    save_feature_importance(pipeline)

    return accuracy, roc_auc


def save_confusion_matrix(cm: np.ndarray) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{CFG.results_dir}/confusion_matrix.png")
    plt.close()


def save_roc_curve(y_true: pd.Series, y_prob: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(f"{CFG.results_dir}/roc_curve.png")
    plt.close()


def save_feature_importance(pipeline: Pipeline) -> None:

    model = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    num_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    cat_features = preprocessor.named_transformers_["cat"] \
        .named_steps["encoder"] \
        .get_feature_names_out(["Sex", "Embarked"])

    feature_names = list(num_features) + list(cat_features)
    coefficients = model.coef_[0]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients
    }).sort_values(by="Coefficient", ascending=False)

    importance_df.to_csv(f"{CFG.results_dir}/feature_importance.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x="Coefficient", y="Feature")
    plt.tight_layout()
    plt.savefig(f"{CFG.results_dir}/feature_importance.png")
    plt.close()


# =====================================================
# Training Logic
# =====================================================

def train_model(train_df: pd.DataFrame) -> Pipeline:

    X = train_df.drop("Survived", axis=1)
    y = train_df["Survived"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=CFG.test_size,
        stratify=y,
        random_state=CFG.random_state
    )

    pipeline = build_pipeline(CFG.random_state)

    logging.info("Performing cross-validation...")
    cv = StratifiedKFold(
        n_splits=CFG.cv_folds,
        shuffle=True,
        random_state=CFG.random_state
    )

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    logging.info(f"CV Accuracy Mean: {cv_scores.mean():.4f}")
    logging.info(f"CV Accuracy Std: {cv_scores.std():.4f}")

    logging.info("Training final model...")
    pipeline.fit(X_train, y_train)

    evaluate_model(pipeline, X_val, y_val)

    return pipeline





# =====================================================
# Main Execution
# =====================================================

def main() -> None:

    train_df = basic_cleaning(load_data(CFG.train_path))
    test_df = basic_cleaning(load_data(CFG.test_path))

    pipeline = train_model(train_df)

    logging.info("Saving trained model...")
    joblib.dump(pipeline, CFG.model_path)

    logging.info("Generating Kaggle submission...")
    predictions = pipeline.predict(test_df).astype(int)

    submission = pd.DataFrame({
        "PassengerId": load_data(CFG.test_path)["PassengerId"],
        "Survived": predictions
    })

    submission.to_csv(f"{CFG.results_dir}/submission.csv", index=False)

    logging.info("All artifacts saved successfully.")


if __name__ == "__main__":
    main()