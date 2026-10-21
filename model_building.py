"""
model_building.py
Train and evaluate employee attrition classifiers, then save the best pipeline.

Run:
    pip install -r requirements.txt
    python model_building.py --data "Palo Alto Networks (1).csv"

Target convention: Attrition=1 means left; Attrition=0 means stayed.
This is a demonstration / workforce-planning aid, not a decision system for
employment actions. Review predictions with HR context and validate before use.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

RANDOM_STATE = 42
TARGET = "Attrition"
ARTIFACT_PATH = Path("attrition_model.joblib")
METRICS_PATH = Path("model_metrics.json")


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create requested features while preserving a DataFrame interface."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        # Numeric conversion protects against blanks / imported text values.
        for col in frame.columns:
            if col not in ("BusinessTravel", "Department", "EducationField",
                           "Gender", "JobRole", "MaritalStatus", "OverTime"):
                frame[col] = pd.to_numeric(frame[col], errors="coerce")

        income = pd.to_numeric(frame.get("MonthlyIncome", 0), errors="coerce")
        experience = pd.to_numeric(frame.get("TotalWorkingYears", 0), errors="coerce")
        frame["IncomeToExperienceRatio"] = income / (experience.clip(lower=0) + 1)
        years_company = pd.to_numeric(frame.get("YearsAtCompany", 0), errors="coerce")
        years_promotion = pd.to_numeric(frame.get("YearsSinceLastPromotion", 0), errors="coerce")
        frame["PromotionDelayFlag"] = (
            (years_company >= 3) & (years_promotion >= 3)
        ).astype(int)
        satisfaction_cols = [
            c for c in ["JobInvolvement", "JobSatisfaction", "EnvironmentSatisfaction",
                        "RelationshipSatisfaction", "WorkLifeBalance"]
            if c in frame.columns
        ]
        if satisfaction_cols:
            frame["EngagementComposite"] = frame[satisfaction_cols].apply(
                pd.to_numeric, errors="coerce"
            ).mean(axis=1)
        else:
            frame["EngagementComposite"] = np.nan
        overtime = frame.get("OverTime", pd.Series("", index=frame.index)).astype(str).str.lower()
        distance = pd.to_numeric(frame.get("DistanceFromHome", np.nan), errors="coerce")
        frame["WorkloadStressFlag"] = (
            overtime.isin(["yes", "true", "1"]) |
            (pd.to_numeric(frame.get("WorkLifeBalance", np.nan), errors="coerce") <= 2)
        ).astype(int)
        # Keep ratios finite; imputer handles remaining missing values.
        frame["IncomeToExperienceRatio"] = frame["IncomeToExperienceRatio"].replace(
            [np.inf, -np.inf], np.nan
        )
        return frame


def build_pipeline(X: pd.DataFrame, estimator):
    # Fit the column selector against the engineered schema so the four
    # added features are actually passed into preprocessing and the model.
    engineered = FeatureEngineer().fit_transform(X)
    numeric_cols = engineered.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in engineered.columns if c not in numeric_cols]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ])
    return Pipeline([
        ("feature_engineering", FeatureEngineer()),
        ("preprocess", preprocess),
        ("model", estimator),
    ])


def evaluate_model(name, pipeline, X_test, y_test):
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, target_names=["Stayed (0)", "Left (1)"],
            output_dict=True, zero_division=0
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Build employee attrition prediction models.")
    parser.add_argument("--data", default="Palo Alto Networks (1).csv",
                        help="Path to input CSV (default: Palo Alto Networks (1).csv)")
    parser.add_argument("--artifact", default=str(ARTIFACT_PATH),
                        help="Where to save the trained model artifact")
    parser.add_argument("--metrics", default=str(METRICS_PATH),
                        help="Where to save evaluation metrics JSON")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path.resolve()}. Put the CSV beside this script "
            "or pass its path using --data."
        )

    df = pd.read_csv(data_path)
    if TARGET not in df.columns:
        raise ValueError(f"Dataset must contain target column '{TARGET}'.")
    df = df.drop_duplicates().copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].isin([0, 1])].copy()
    if df.empty:
        raise ValueError("No valid rows found: Attrition must be coded 0/1.")
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET]).copy()

    # Remove columns that are entirely empty; imputers cannot learn from them.
    X = X.dropna(axis=1, how="all")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=2,
            random_state=RANDOM_STATE
        ),
    }

    results = []
    fitted = {}
    for name, estimator in candidates.items():
        pipeline = build_pipeline(X_train, estimator)
        if name == "Gradient Boosting":
            # GradientBoostingClassifier has no class_weight parameter; use balanced
            # per-row weights calculated from the training labels only.
            sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
            pipeline.fit(X_train, y_train, model__sample_weight=sample_weights)
        else:
            pipeline.fit(X_train, y_train)
        result = evaluate_model(name, pipeline, X_test, y_test)
        results.append(result)
        fitted[name] = pipeline
        print(f"{name:20s} | ROC-AUC={result['roc_auc']:.3f} "
              f"| F1={result['f1_score']:.3f} | Recall={result['recall']:.3f}")

    # ROC-AUC is the primary selection metric; F1 breaks ties.
    best = max(results, key=lambda r: (r["roc_auc"], r["f1_score"]))
    best_name = best["model"]
    artifact = {
        "pipeline": fitted[best_name],
        "model_name": best_name,
        "metrics": results,
        "selected_metrics": best,
        "input_columns": X.columns.tolist(),
        "target": TARGET,
        "risk_thresholds": {"low_upper_exclusive": 0.30, "medium_upper_exclusive": 0.60},
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "positive_rate": float(y.mean()),
        "dataset_name": data_path.name,
    }
    joblib.dump(artifact, args.artifact)
    with open(args.metrics, "w", encoding="utf-8") as f:
        json.dump({"selected_model": best_name, "results": results}, f, indent=2)

    print(f"\nSelected model: {best_name}")
    print(f"Saved model artifact: {Path(args.artifact).resolve()}")
    print(f"Saved metrics: {Path(args.metrics).resolve()}")
    print("\nReminder: evaluate model fairness, calibration, drift, and business value "
          "before using predictions in real HR workflows.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
