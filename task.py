# task.py - ML Workflow for Sportsbet App with Linear Regression & XGBoost
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb  # XGBoost

def train_model(X, y, model_type="linear", test_size=0.3, random_state=123):
    """
    Train a machine learning model on X, y.

    Parameters:
        X (DataFrame): Feature matrix
        y (Series): Target variable
        model_type (str): "linear" for LinearRegression, "xgb" for XGBoost
        test_size (float): Fraction of data for test set
        random_state (int): Random seed for reproducibility

    Returns:
        model: trained ML object
        X_train, X_test, y_train, y_test, y_pred
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if model_type == "linear":
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    elif model_type == "xgb":
        model = xgb.XGBRegressor(
            objective='reg:squarederror', random_state=random_state
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    else:
        raise ValueError("model_type must be 'linear' or 'xgb'")

    return model, X_train, X_test, y_train, y_test, y_pred

def calculate_metrics(y_test, y_pred):
    """
    Calculate R² and RMSE for predictions.

    Parameters:
        y_test (array-like): True target values
        y_pred (array-like): Predicted target values

    Returns:
        r2 (float), rmse (float)
    """
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return r2, rmse

def plot_predicted_vs_actual(y_test, y_pred):
    """
    Create a scatter plot of predicted vs actual values with a perfect-fit line.

    Parameters:
        y_test (array-like): True target values
        y_pred (array-like): Predicted target values

    Returns:
        matplotlib.pyplot object
    """
    plt.figure(figsize=(6,6))
    sns.set(style="whitegrid")
    sns.scatterplot(x=y_test, y=y_pred, color="blue", s=60)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    plt.xlabel("Actual", fontsize=12)
    plt.ylabel("Predicted", fontsize=12)
    plt.title("Predicted vs Actual", fontsize=14)
    plt.tight_layout()
    return plt
