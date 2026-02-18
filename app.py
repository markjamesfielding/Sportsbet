# app.py - Clean Modular Streamlit ML App
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import task

st.set_page_config(page_title="Sportsbet ML Task", layout="wide")

# ---- App Title ----
st.title("Sportsbet ML Task - Modular Template")

st.markdown("""
This app allows you to:

• Upload a CSV dataset **or** use a built-in sample dataset  
• Select a target variable and features  
• Fit a regression model  
• View **R²**, **RMSE**, and a Predicted vs Actual plot  
• Download the ML workflow code  

---
""")

# ---- STEP 1: Data Source Selection ----
st.subheader("Step 1: Choose Data Source")

data_option = st.radio(
    "Select one option:",
    ("Use Sample Dataset", "Upload Your Own CSV")
)

df = None

# ---- SAMPLE DATA OPTION ----
if data_option == "Use Sample Dataset":
    np.random.seed(123)
    df = pd.DataFrame({
        "Feature1": np.random.rand(50) * 10,
        "Feature2": np.random.rand(50) * 5,
        "Feature3": np.random.rand(50) * 20,
    })
    df["Target"] = (
        2.5 * df["Feature1"]
        - 1.5 * df["Feature2"]
        + 0.5 * df["Feature3"]
        + np.random.randn(50) * 2
    )

    st.success("Sample dataset loaded.")
    st.dataframe(df.head())

# ---- FILE UPLOAD OPTION ----
elif data_option == "Upload Your Own CSV":
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Dataset uploaded successfully.")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading file: {e}")

# ---- STEP 2: Model Configuration ----
if df is not None:

    st.subheader("Step 2: Configure Model")

    target = st.selectbox("Select Target Variable", df.columns)

    features = st.multiselect(
        "Select Feature Columns",
        [col for col in df.columns if col != target]
    )

    model_type = st.radio(
        "Select Model Type",
        ("linear", "xgb"),
        index=0,
        help="Linear Regression or XGBoost"
    )

    # ---- RUN MODEL ----
    if st.button("Run Model"):

        if not features:
            st.warning("Please select at least one feature.")
        else:
            X = df[features]
            y = df[target]

            with st.spinner("Training model..."):
                # Call task.py functions
                model, X_train, X_test, y_train, y_test, y_pred = task.train_model(
                    X, y, model_type=model_type
                )
                r2, rmse = task.calculate_metrics(y_test, y_pred)

            st.subheader("Model Performance")
            st.metric("R²", f"{r2:.3f}")
            st.metric("RMSE", f"{rmse:.3f}")

            st.subheader("Predicted vs Actual")
            fig = task.plot_predicted_vs_actual(y_test, y_pred)
            st.pyplot(fig)

# ---- STEP 3: Download ML Workflow Code ----
st.subheader("Download the ML Workflow Code")

try:
    with open("task.py", "r", encoding="utf-8") as f:
        code_text = f.read()

    with st.expander("Show / Hide Python ML Code"):
        st.code(code_text, language="python")
        st.download_button(
            label="Download task.py",
            data=code_text,
            file_name="task.py",
            mime="text/plain"
        )

except Exception as e:
    st.error(f"Could not display or download the code: {e}")
