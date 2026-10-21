"""
app.py — Streamlit employee attrition risk dashboard.

Run after training:
    streamlit run app.py

Expected files in the same directory:
    attrition_model.joblib
    Palo Alto Networks (1).csv  (or set DATA_PATH in the sidebar)
"""
from __future__ import annotations

from pathlib import Path
import warnings

import joblib
from model_building import FeatureEngineer
import numpy as np
from model_building import FeatureEngineer
import pandas as pd
from model_building import FeatureEngineer
import streamlit as st
from model_building import FeatureEngineer
import plotly.express as px
from model_building import FeatureEngineer

st.set_page_config(page_title="Employee Attrition Risk", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

DEFAULT_DATA = "Palo Alto Networks (1).csv"
DEFAULT_MODEL = "attrition_model.joblib"


@st.cache_resource
def load_artifact(path: str):
    return joblib.load(path)


@st.cache_data
def load_data(path: str):
    return pd.read_csv(path)


def risk_category(probability: float) -> str:
    if probability < 0.30:
        return "Low Risk"
    if probability <= 0.60:
        return "Medium Risk"
    return "High Risk"


def risk_recommendations(row: pd.Series) -> list[str]:
    """Non-prescriptive, reviewable suggestions based on common workforce signals."""
    suggestions = []
    if str(row.get("OverTime", "")).strip().lower() == "yes":
        suggestions.append("Discuss workload, overtime frequency, and capacity planning.")
    if pd.to_numeric(pd.Series([row.get("JobSatisfaction", np.nan)]), errors="coerce").iloc[0] <= 2:
        suggestions.append("Offer a confidential check-in about role fit and job satisfaction.")
    if pd.to_numeric(pd.Series([row.get("EnvironmentSatisfaction", np.nan)]), errors="coerce").iloc[0] <= 2:
        suggestions.append("Ask about team environment, tools, and workplace support.")
    if pd.to_numeric(pd.Series([row.get("WorkLifeBalance", np.nan)]), errors="coerce").iloc[0] <= 2:
        suggestions.append("Explore work-life balance and flexible-work options where feasible.")
    years = pd.to_numeric(pd.Series([row.get("YearsSinceLastPromotion", np.nan)]), errors="coerce").iloc[0]
    tenure = pd.to_numeric(pd.Series([row.get("YearsAtCompany", np.nan)]), errors="coerce").iloc[0]
    if pd.notna(years) and pd.notna(tenure) and tenure >= 3 and years >= 3:
        suggestions.append("Review career development, promotion criteria, and growth opportunities.")
    if not suggestions:
        suggestions.append("Arrange a voluntary stay conversation and review career goals and support needs.")
    return suggestions


def top_global_features(artifact, limit=10):
    """Return readable feature importance when supported by the selected estimator."""
    pipeline = artifact["pipeline"]
    try:
        pre = pipeline.named_steps["preprocess"]
        model = pipeline.named_steps["model"]
        names = pre.get_feature_names_out()
        if hasattr(model, "feature_importances_"):
            values = model.feature_importances_
        elif hasattr(model, "coef_"):
            values = np.abs(np.asarray(model.coef_)[0])
        else:
            return pd.DataFrame(columns=["Feature", "Importance"])
        out = pd.DataFrame({"Feature": names, "Importance": values})
        out["Feature"] = out["Feature"].str.replace(r"^(numeric|categorical)__", "", regex=True)
        out["Feature"] = out["Feature"].str.replace(r"^onehot__", "", regex=True)
        out = out.groupby("Feature", as_index=False)["Importance"].sum()
        return out.sort_values("Importance", ascending=False).head(limit)
    except Exception:
        return pd.DataFrame(columns=["Feature", "Importance"])


def predict_probabilities(artifact, frame):
    # Model pipeline itself creates the engineered fields.
    return artifact["pipeline"].predict_proba(frame)[:, 1]


st.title("📊 Employee Attrition Prediction & Risk Scoring")
st.caption("Decision-support prototype for workforce planning. Predictions are estimates, not facts or recommendations to take adverse employment action.")

with st.sidebar:
    st.header("Configuration")
    data_path = st.text_input("Dataset CSV path", value=DEFAULT_DATA)
    model_path = st.text_input("Trained model path", value=DEFAULT_MODEL)
    st.markdown("---")
    st.markdown("**Risk bands**")
    st.markdown("- Low: probability below 30%")
    st.markdown("- Medium: 30% to 60%")
    st.markdown("- High: above 60%")
    st.markdown("---")
    st.caption("Use only with appropriate access controls, employee notice, and HR/legal review.")

if not Path(model_path).exists():
    st.error(f"Model artifact not found: `{model_path}`")
    st.info("Run `python model_building.py --data \"Palo Alto Networks (1).csv\"` first.")
    st.stop()
if not Path(data_path).exists():
    st.error(f"Dataset not found: `{data_path}`. Update the path in the sidebar.")
    st.stop()

try:
    artifact = load_artifact(model_path)
    raw = load_data(data_path)
except Exception as exc:
    st.error(f"Could not load model or data: {exc}")
    st.stop()

if "Attrition" in raw.columns:
    # Exclude the known outcome from predictors; the app displays predicted risk only.
    employee_data = raw.drop(columns=["Attrition"]).copy()
else:
    employee_data = raw.copy()

# Keep a stable row index for employee selection; dataset contains no unique employee ID.
employee_data = employee_data.reset_index(drop=True)
employee_data.insert(0, "Employee Record", [f"Record {i+1:04d}" for i in range(len(employee_data))])

try:
    employee_data["Attrition Probability"] = predict_probabilities(
        artifact, employee_data.drop(columns=["Employee Record"])
    )
    employee_data["Risk Category"] = employee_data["Attrition Probability"].map(risk_category)
except Exception as exc:
    st.error(f"Prediction failed. Confirm that the CSV matches the model's training columns. Details: {exc}")
    st.stop()

# Use all available rows as a demo dashboard. This CSV is a historical dataset, not a live HR feed.
with st.container(border=True):
    st.subheader("Attrition Risk Dashboard")
    total = len(employee_data)
    high_count = int((employee_data["Risk Category"] == "High Risk").sum())
    medium_count = int((employee_data["Risk Category"] == "Medium Risk").sum())
    low_count = int((employee_data["Risk Category"] == "Low Risk").sum())
    avg_risk = employee_data["Attrition Probability"].mean() if total else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Employee records", f"{total:,}")
    c2.metric("High-risk records", f"{high_count:,}", f"{high_count / max(total, 1):.1%} of records")
    c3.metric("Average predicted risk", f"{avg_risk:.1%}")
    c4.metric("Selected model", artifact.get("model_name", "Unknown"))

    left, right = st.columns(2)
    risk_order = ["Low Risk", "Medium Risk", "High Risk"]
    risk_counts = employee_data["Risk Category"].value_counts().reindex(risk_order, fill_value=0).rename_axis("Risk Category").reset_index(name="Employees")
    with left:
        fig = px.pie(risk_counts, names="Risk Category", values="Employees",
                     title="Risk distribution", hole=0.45)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        dept_col = "Department" if "Department" in employee_data.columns else None
        if dept_col:
            dept_risk = employee_data.groupby(["Department", "Risk Category"]).size().reset_index(name="Employees")
            fig = px.bar(dept_risk, x="Department", y="Employees", color="Risk Category",
                         barmode="stack", title="Risk categories by department")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Department column not present in this dataset.")

with st.container(border=True):
    st.subheader("Department-Level Risk View & Filters")
    f1, f2, f3 = st.columns([1, 1, 1.2])
    departments = sorted(employee_data["Department"].dropna().astype(str).unique()) if "Department" in employee_data else []
    roles = sorted(employee_data["JobRole"].dropna().astype(str).unique()) if "JobRole" in employee_data else []
    with f1:
        selected_departments = st.multiselect("Department", departments, default=departments)
    with f2:
        selected_roles = st.multiselect("Job role", roles, default=roles)
    with f3:
        min_risk, max_risk = st.slider("Predicted risk range", 0, 100, (0, 100), 5, format="%d%%")
    filtered = employee_data.copy()
    if "Department" in filtered:
        filtered = filtered[filtered["Department"].astype(str).isin(selected_departments)]
    if "JobRole" in filtered:
        filtered = filtered[filtered["JobRole"].astype(str).isin(selected_roles)]
    filtered = filtered[
        filtered["Attrition Probability"].between(min_risk / 100, max_risk / 100)
    ].sort_values("Attrition Probability", ascending=False)
    display_cols = ["Employee Record"]
    for col in ["Department", "JobRole", "Age", "YearsAtCompany", "OverTime", "Attrition Probability", "Risk Category"]:
        if col in filtered.columns and col not in display_cols:
            display_cols.append(col)
    table = filtered[display_cols].copy()
    if "Attrition Probability" in table:
        table["Attrition Probability"] = table["Attrition Probability"].map(lambda x: f"{x:.1%}")
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered risk view (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="employee_attrition_risk_view.csv",
        mime="text/csv",
    )

with st.container(border=True):
    st.subheader("Employee Risk Profile")
    if filtered.empty:
        st.warning("No employee records match the selected filters. Widen the filters to select a record.")
    else:
        choices = filtered["Employee Record"].tolist()
        selected_record = st.selectbox("Select employee record", choices)
        selected = employee_data.loc[employee_data["Employee Record"] == selected_record].iloc[0]
        probability = float(selected["Attrition Probability"])
        category = selected["Risk Category"]
        p1, p2, p3 = st.columns(3)
        p1.metric("Predicted attrition probability", f"{probability:.1%}")
        p2.metric("Risk category", category)
        p3.metric("Department / role", f"{selected.get('Department', 'N/A')} / {selected.get('JobRole', 'N/A')}")
        st.progress(min(max(probability, 0.0), 1.0), text=f"Estimated risk: {probability:.1%}")

        with st.expander("Employee attributes used for context", expanded=True):
            preferred = [
                "Age", "Gender", "Department", "JobRole", "JobLevel", "YearsAtCompany",
                "TotalWorkingYears", "YearsInCurrentRole", "YearsSinceLastPromotion",
                "YearsWithCurrManager", "OverTime", "MonthlyIncome", "JobSatisfaction",
                "EnvironmentSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance",
                "JobInvolvement", "BusinessTravel", "DistanceFromHome"
            ]
            available = [c for c in preferred if c in selected.index]
            st.dataframe(pd.DataFrame({"Attribute": available, "Value": [selected[c] for c in available]}),
                         use_container_width=True, hide_index=True)

        st.markdown("**Suggested conversation topics**")
        for suggestion in risk_recommendations(selected):
            st.markdown(f"- {suggestion}")
        st.caption("These are prompts for voluntary, supportive conversations—not automated interventions or conclusions about an individual.")

with st.container(border=True):
    st.subheader("Model Evaluation")
    metrics = artifact.get("metrics", [])
    if metrics:
        metrics_df = pd.DataFrame(metrics)
        cols = [c for c in ["model", "accuracy", "precision", "recall", "f1_score", "roc_auc"] if c in metrics_df]
        shown = metrics_df[cols].copy()
        for col in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            if col in shown:
                shown[col] = shown[col].map(lambda x: f"{x:.3f}")
        st.dataframe(shown, use_container_width=True, hide_index=True)
        st.caption("Metrics are from a held-out stratified test set. Choose the model based on business costs and validation—not accuracy alone.")
    else:
        st.info("No evaluation metrics were found in the model artifact.")

    importance = top_global_features(artifact)
    if not importance.empty:
        st.markdown("**Global model feature importance (not individual causal explanations)**")
        fig = px.bar(importance.sort_values("Importance"), x="Importance", y="Feature",
                     orientation="h", title="Most influential model inputs")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Privacy & responsible use: This prototype uses a historical sample dataset and does not establish causal reasons for attrition. Limit access to authorized staff, avoid using protected characteristics for employment decisions, assess disparate impact, calibrate probabilities, and obtain appropriate HR/legal/privacy review before operational deployment.")
