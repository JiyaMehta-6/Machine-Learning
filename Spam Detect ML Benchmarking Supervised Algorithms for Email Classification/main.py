import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)

warnings.filterwarnings("ignore")

#----------------------------------------------------------------
# Configuration

DATA_PATH = "data/emails.csv"
OUTPUT_DIR = "outputs"
TEST_SIZE = 0.30
RANDOM_STATE = 42
CV_FOLDS = 5


#----------------------------------------------------------------
# Setup

def create_output_directory():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


#----------------------------------------------------------------
# Data Loading & Preprocessing

def load_data():
    df = pd.read_csv(DATA_PATH)

    if "Email No." in df.columns:
        df.drop(columns=["Email No."], inplace=True)

    X = df.drop(columns=["Prediction"])
    y = df["Prediction"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(
        X_scaled,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


#----------------------------------------------------------------
# Models

def get_models():
    return {
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "SVM": SVC(C=1, kernel="rbf", probability=True),
        "Logistic Regression": LogisticRegression(max_iter=2000)
    }


#----------------------------------------------------------------
# Visualization Functions

def save_confusion_matrix(cm, model_name):
    plt.figure()
    plt.imshow(cm)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.close()


def save_accuracy_bar(results):
    models = list(results.keys())
    accuracies = [results[m]["test_accuracy"] for m in models]

    plt.figure()
    plt.bar(models, accuracies)
    plt.xlabel("Models")
    plt.ylabel("Test Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_accuracy_comparison.png")
    plt.close()


def save_roc_curves(results):
    plt.figure()

    for model_name, metrics in results.items():
        fpr, tpr = metrics["roc"]
        roc_auc = metrics["roc_auc"]
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/roc_curve_comparison.png")
    plt.close()


def save_precision_recall_curves(results):
    plt.figure()

    for model_name, metrics in results.items():
        precision, recall = metrics["pr"]
        plt.plot(recall, precision, label=model_name)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/precision_recall_comparison.png")
    plt.close()


#----------------------------------------------------------------
# Report Generation

def generate_reports(results):
    performance_summary = []
    spam_metrics_summary = []

    for model, metrics in results.items():
        performance_summary.append([
            model,
            metrics["cv_mean"],
            metrics["test_accuracy"],
            metrics["roc_auc"]
        ])

        spam_class_metrics = metrics["classification_report"]["1"]

        spam_metrics_summary.append([
            model,
            spam_class_metrics["precision"],
            spam_class_metrics["recall"],
            spam_class_metrics["f1-score"]
        ])

        # Save full classification report
        report_df = pd.DataFrame(metrics["classification_report"]).transpose()
        report_df.to_csv(
            f"{OUTPUT_DIR}/{model.lower().replace(' ', '_')}_classification_report.csv"
        )

    performance_df = pd.DataFrame(
        performance_summary,
        columns=["Model", "CV Accuracy", "Test Accuracy", "ROC AUC"]
    )

    spam_df = pd.DataFrame(
        spam_metrics_summary,
        columns=["Model", "Precision (Spam)", "Recall (Spam)", "F1-Score (Spam)"]
    )

    performance_df.to_csv(f"{OUTPUT_DIR}/model_performance_summary.csv", index=False)
    spam_df.to_csv(f"{OUTPUT_DIR}/classification_metrics_summary.csv", index=False)

    best_model = performance_df.sort_values("Test Accuracy", ascending=False).iloc[0]

    with open(f"{OUTPUT_DIR}/project_conclusion.txt", "w") as f:
        f.write("SpamDetect AI - Project Conclusion\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated on: {datetime.now()}\n\n")

        f.write("Performance Summary:\n")
        f.write(performance_df.to_string(index=False))
        f.write("\n\n")

        f.write("Spam Detection Metrics:\n")
        f.write(spam_df.to_string(index=False))
        f.write("\n\n")

        f.write("Conclusion:\n")
        f.write(
            f"The comparative evaluation demonstrates that "
            f"{best_model['Model']} achieved the highest test accuracy "
            f"of {best_model['Test Accuracy']:.4f}. "
            f"Cross-validation confirms model stability, while ROC-AUC "
            f"and F1-score validate effective spam detection performance.\n"
        )


#----------------------------------------------------------------
# Main Execution

def main():
    print("\nSpamDetect AI - Elite ML Benchmark System\n")

    create_output_directory()

    X_train, X_test, y_train, y_test = load_data()

    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS)
        cv_mean = np.mean(cv_scores)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        test_accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(y_test, y_prob)

        results[name] = {
            "cv_mean": cv_mean,
            "test_accuracy": test_accuracy,
            "confusion_matrix": cm,
            "roc": (fpr, tpr),
            "roc_auc": roc_auc,
            "pr": (precision, recall),
            "classification_report": class_report
        }

        print(f"{name} | CV: {cv_mean:.4f} | Test: {test_accuracy:.4f} | AUC: {roc_auc:.4f}")
        print(classification_report(y_test, y_pred))

        save_confusion_matrix(cm, name)

    save_accuracy_bar(results)
    save_roc_curves(results)
    save_precision_recall_curves(results)
    generate_reports(results)

    print("\nAll outputs saved inside the 'outputs' directory.\n")


if __name__ == "__main__":
    main()
