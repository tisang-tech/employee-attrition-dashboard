import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Wedge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Employee Attrition Analysis System", layout="wide", page_icon="👥")
colors = ["#1f77b4", "#d62728"]

@st.cache_data
def load_data():
    return pd.read_csv("employee_attrition.csv")

@st.cache_resource
def train_models(_df):
    X = _df.drop(columns=["Employee_ID", "Attrition"])
    y = (_df["Attrition"] == "Yes").astype(int)
    X_encoded = pd.get_dummies(X, drop_first=True)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_encoded), columns=X_encoded.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42),
    }
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results.append({
            "Model": name,
            "Accuracy (%)": round(accuracy_score(y_test, pred) * 100, 2),
            "Precision": round(precision_score(y_test, pred), 2),
            "Recall": round(recall_score(y_test, pred), 2),
            "F1-score": round(f1_score(y_test, pred), 2),
        })
    results_df = pd.DataFrame(results).sort_values("F1-score", ascending=False).reset_index(drop=True)
    best_name = results_df.loc[0, "Model"]
    best_model = models[best_name]
    conf = confusion_matrix(y_test, best_model.predict(X_test))
    return best_name, best_model, results_df, conf, scaler, list(X_encoded.columns)

def risk_gauge(prob):
    fig, ax = plt.subplots(figsize=(5, 3))
    for start, end, color in [(126, 180, "lightgreen"), (54, 126, "gold"), (0, 54, "salmon")]:
        ax.add_patch(Wedge((0.5, 0.5), 0.4, start, end, facecolor=color, alpha=0.85, width=0.15))
    angle = np.radians(180 - prob * 180)
    ax.annotate("", xy=(0.5 + 0.32 * np.cos(angle), 0.5 + 0.32 * np.sin(angle)),
                xytext=(0.5, 0.5), arrowprops=dict(arrowstyle="->", color="black", lw=3))
    ax.text(0.5, 0.42, f"{prob * 100:.1f}%", ha="center", fontsize=16, fontweight="bold")
    ax.set_title("Attrition Risk Score")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.35, 0.95)
    return fig

df = load_data()
best_name, best_model, results_df, conf, scaler, feature_cols = train_models(df)

st.title("Employee Attrition Analysis System")
tab1, tab2, tab3 = st.tabs(["KPI & Trends", "Model Performance", "What-If Predictor"])

with tab1:
    total = len(df)
    left = int((df["Attrition"] == "Yes").sum())
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Workforce", f"{total:,}")
    k2.metric("Retained Staff", f"{total - left:,}")
    k3.metric("Attrited Staff", f"{left:,}")
    k4.metric("Overall Attrition Rate", f"{(left / total * 100):.1f}%")

    flag = (df["Attrition"] == "Yes").astype(int)
    groups = {
        "Age Group": pd.cut(df["Age"], [17, 25, 35, 45, 60], labels=["<=25", "26-35", "36-45", "46+"]),
        "Tenure (Years at Company)": pd.cut(df["Years_At_Company"], [-1, 2, 5, 10, 40], labels=["0-2", "3-5", "6-10", "10+"]),
        "Department": df["Department"],
        "Job Satisfaction": df["Job_Satisfaction"],
    }
    c1, c2 = st.columns(2)
    for i, (title, group) in enumerate(groups.items()):
        rate = flag.groupby(group, observed=True).mean() * 100
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        sns.barplot(ax=ax, x=rate.index.astype(str), y=rate.values, color="#d62728")
        ax.set_title("Attrition Rate by " + title)
        ax.set_xlabel(title)
        ax.set_ylabel("Attrition Rate (%)")
        ax.bar_label(ax.containers[0], fmt="%.1f%%")
        plt.tight_layout()
        (c1 if i % 2 == 0 else c2).pyplot(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    sns.countplot(ax=axes[0], data=df, x="OverTime", hue="Attrition", palette=colors)
    axes[0].set_title("Attrition vs OverTime")
    sns.boxplot(ax=axes[1], data=df, x="Attrition", y="Monthly_Income", hue="Attrition", palette=colors, legend=False)
    axes[1].set_title("Monthly Income by Attrition")
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.header("Machine Learning Models")
    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("Model Comparison")
        st.dataframe(results_df)
        st.success(f"Best Model (Highest F1-score): {best_name}")
    with right_col:
        st.subheader(f"Confusion Matrix ({best_name})")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(conf, annot=True, fmt="d", cmap="Blues", xticklabels=["Retained", "Attrited"], yticklabels=["Retained", "Attrited"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    st.subheader("Top 10 Drivers of Attrition")
    if hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_
    else:
        imp = np.abs(best_model.coef_[0])
    top = pd.Series(imp, index=feature_cols).sort_values().tail(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top.index, top.values, color="#7ec8f8")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    st.header("Interactive What-If Scenario Builder")
    with st.form("what_if"):
        a, b, c = st.columns(3)
        age = a.number_input("Age", 20, 60, 29)
        dept = b.selectbox("Department", sorted(df["Department"].unique()))
        wlb = c.slider("Work-Life Balance (1-4)", 1, 4, 2)
        dist = a.number_input("Distance From Home (km)", 1, 29, 15)
        overtime = b.selectbox("OverTime", ["Yes", "No"])
        years = c.number_input("Years At Company", 0, 30, 2)
        income = a.number_input("Monthly Income ($)", 2000, 20000, 5600, step=100)
        job_sat = b.slider("Job Satisfaction (1-4)", 1, 4, 2)
        travel = c.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"], index=1)
        submitted = st.form_submit_button("Run Prediction")

    if submitted:
        profile = {}
        for col in df.drop(columns=["Employee_ID", "Attrition"]).columns:
            profile[col] = df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else df[col].mode()[0]
        profile.update({
            "Age": age, "Department": dept, "Work_Life_Balance": wlb,
            "Distance_From_Home": dist, "OverTime": overtime, "Years_At_Company": years,
            "Monthly_Income": income, "Job_Satisfaction": job_sat, "Business_Travel": travel
        })
        profile["Job_Role"] = df[df["Department"] == dept]["Job_Role"].mode()[0]

        emp = pd.get_dummies(pd.DataFrame([profile])).reindex(columns=feature_cols, fill_value=0).astype(float)
        emp_scaled = pd.DataFrame(scaler.transform(emp), columns=feature_cols)
        prob = best_model.predict_proba(emp_scaled)[0][1]

        st.pyplot(risk_gauge(prob))
        if prob > 0.7:
            st.error("HIGH RISK: Immediate retention measures required.")
        elif prob > 0.3:
            st.warning("MEDIUM RISK: Monitor employee engagement.")
        else:
            st.success("LOW RISK: Employee is likely to stay.")
