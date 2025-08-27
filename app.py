import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from dotenv import load_dotenv
import os
import google.generativeai as genai
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.tree import DecisionTreeClassifier

# -----------------------------
# Load Gemini API Key
# -----------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def ask_gemini(prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

# -----------------------------
# Helper: Make column names unique
# -----------------------------
def make_unique(cols):
    seen = {}
    new_cols = []
    for col in cols:
        if col not in seen:
            seen[col] = 1
            new_cols.append(col)
        else:
            new_name = f"{col}_{seen[col]}"
            seen[col] += 1
            new_cols.append(new_name)
    return new_cols

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🤖 Dataset Analyser with ML Models")
st.write("Upload a dataset and explore insights, ML suggestions, and AI guidance!")

uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Make columns unique
    df.columns = make_unique(df.columns)

    # Create Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Dataset Overview",
        "Visualizations",
        "AI Insights",
        "ML Prototyping",
        "Feature Engineering",
        "Project Ideas",
        "AI Tutor"
    ])

    # -----------------------------
    # Tab 1: Dataset Overview
    # -----------------------------
    with tab1:
        st.subheader("📋 Dataset Overview")
        st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
        st.dataframe(df.head())
        st.write("Missing Values:")
        st.write(df.isnull().sum())
        st.write("Data Types:")
        st.write(df.dtypes)

    # -----------------------------
    # Tab 2: Visualizations
    # -----------------------------
    with tab2:
        st.subheader("📈 Interactive Visualizations")
        cols = df.columns.tolist()

        x_axis = st.selectbox("Select X-axis", ["--Select--"] + cols, index=0, key="xaxis")
        y_axis = st.selectbox("Select Y-axis", ["--Select--"] + cols, index=0, key="yaxis")

        chart_type = st.radio("Chart Type", ["Scatter", "Histogram", "Boxplot"], key="chart_type")

        if x_axis != "--Select--" and y_axis != "--Select--":
            if chart_type == "Scatter":
                fig = px.scatter(df, x=x_axis, y=y_axis, trendline="ols")
            elif chart_type == "Histogram":
                fig = px.histogram(df, x=x_axis, color=y_axis)
            else:
                fig = px.box(df, x=x_axis, y=y_axis)
            st.plotly_chart(fig)

        elif x_axis != "--Select--" and chart_type in ["Histogram", "Boxplot"]:
            fig = px.histogram(df, x=x_axis) if chart_type=="Histogram" else px.box(df, y=x_axis)
            st.plotly_chart(fig)

        else:
            st.info("Please select columns to visualize.")

    # -----------------------------
    # Tab 3: AI Insights
    # -----------------------------
    with tab3:
        st.subheader("🤖 AI Dataset Insights")
        prompt = f"""
        Analyze the following dataset:
        Columns: {list(df.columns)}
        Data types: {df.dtypes.to_dict()}
        Missing values: {df.isnull().sum().to_dict()}
        Shape: {df.shape}
        """
        if st.button("Generate AI Insights", key="ai_insights"):
            insights = ask_gemini(prompt)
            st.write(insights)

    # -----------------------------
    # Tab 4: ML Prototyping
    # -----------------------------
    with tab4:
        st.subheader("⚡ ML Model Prototyping")
        target_col = st.selectbox("Select Target Column", ["--Select--"] + list(df.columns), key="target")
        if target_col != "--Select--":
            X = df.dropna().drop(columns=[target_col])
            y = df.dropna()[target_col]
            if y.dtype == 'object' or len(y.unique()) < 20:
                problem_type = "classification"
                y = pd.factorize(y)[0]
            else:
                problem_type = "regression"
            st.write(f"Detected Problem Type: **{problem_type}**")

            try:
                X = pd.get_dummies(X)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                if problem_type == "classification":
                    model = DecisionTreeClassifier()
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    acc = accuracy_score(y_test, preds)
                    st.write(f"✅ Baseline Accuracy: {acc:.2f}")
                else:
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    mae = mean_absolute_error(y_test, preds)
                    st.write(f"✅ Baseline MAE: {mae:.2f}")
            except Exception as e:
                st.error(f"Error training model: {e}")

    # -----------------------------
    # Tab 5: Feature Engineering
    # -----------------------------
    with tab5:
        st.subheader("🛠 Feature Engineering Suggestions")
        feature_prompt = f"Suggest feature engineering ideas for this dataset: columns = {list(df.columns)}"
        if st.button("Generate Feature Ideas", key="features"):
            feature_ideas = ask_gemini(feature_prompt)
            st.write(feature_ideas)

    # -----------------------------
    # Tab 6: Project Ideas
    # -----------------------------
    with tab6:
        st.subheader("💡 Real-world ML Project Ideas")
        project_prompt = f"Suggest real-world ML project ideas for dataset with columns: {list(df.columns)}"
        if st.button("Generate Project Ideas", key="projects"):
            project_ideas = ask_gemini(project_prompt)
            st.write(project_ideas)

    # -----------------------------
    # Tab 7: AI Tutor
    # -----------------------------
    with tab7:
        st.subheader("👩‍🏫 AI Tutor Mode")
        user_q = st.text_input("Ask a question about this dataset:", key="tutor")
        if st.button("Ask AI", key="ask_ai") and user_q:
            tutor_prompt = f"Answer this question about the dataset: columns = {list(df.columns)}; Question = {user_q}"
            tutor_answer = ask_gemini(tutor_prompt)
            st.write(tutor_answer)
