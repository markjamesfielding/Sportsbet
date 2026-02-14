# app.py - Modular Streamlit ML App
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import task  # Updated ML workflow module

# ---- App Title ----
st.title("Sportsbet ML Task - Modular Template")

st.markdown("""
This app allows you to upload a CSV dataset, select a target variable and features,
fit a simple linear regression model, and display results including **R²**, **RMSE**, 
and a predicted vs actual scatter plot.

(**The ML workflow code can be downloaded below**)
""")

# ---- Step 0: Optional Sample CSV ----
if st.checkbox("Use Sample Dataset"):
    np.random.seed(123)
    df = pd.DataFrame({
        "Feature1": np.random.rand(50) * 10,
        "Feature2": np.random.rand(50) * 5,
        "Feature3": np.random.rand(50) * 20,
    })
    df["Target"] = 2.5 * df["Feature1"] - 1.5 * df["Feature2"] + 0.5 * df["Feature3"] + np.random.randn(50) * 2
    st.subheader("Sample Data Preview")
    st.dataframe(df.head(10))

# ---- Step 1: Upload CSV ----
uploaded_file = st.file_uploader("Or Upload Your CSV Dataset", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='utf-8')
    st.subheader("Data Preview")
    st.dataframe(df.head(10))

# ---- Step 2: Select target and features ----
if 'df' in locals():
    target = st.selectbox("Select Target Variable", df.columns)
    features = st.multiselect("Select Features", [col for col in df.columns if col != target])

    # ---- Step 3: Run Model Button ----
    if st.button("Run Model") and features:
        X = df[features]
        y = df[target]

        # ---- Train Model and Predict ----
        model, X_train, X_test, y_train, y_test, y_pred = task.train_model(X, y)

        # ---- Metrics ----
        r2, rmse = task.calculate_metrics(y_test, y_pred)
        st.subheader("Metrics")
        st.text(f"R²: {r2:.3f}")
        st.text(f"RMSE: {rmse:.3f}")

        # ---- Predicted vs Actual Plot ----
        st.subheader("Predicted vs Actual")
        plt_fig = task.plot_predicted_vs_actual(y_test, y_pred)
        st.pyplot(plt_fig)

# ---- Step 4: Display / Download the ML code ----
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
