import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Ridge Regression from Scratch", layout="wide")

st.title("📈 Linear Regression with L2 Regularization (From Scratch)")
st.write("Before One-Hot Encoding | Gradient Descent Implementation")

# =========================
# File Upload
# =========================
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Dataset")
    st.dataframe(df.head())

    # =========================
    # Drop NA
    # =========================
    df = df.dropna()
    st.success("NA values dropped")

    # =========================
    # Separate X and y
    # =========================
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)

    # =========================
    # Mean-Max Normalization
    # =========================
    X_mean = X.mean(axis=0)
    X_max = X.max(axis=0)
    X = (X - X_mean) / X_max

    # Add bias
    X = np.c_[np.ones(X.shape[0]), X]

    # =========================
    # Train/Val/Test Split
    # =========================
    m = X.shape[0]
    indices = np.random.permutation(m)

    train_end = int(0.6 * m)
    val_end = int(0.8 * m)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # =========================
    # Sidebar Hyperparameters
    # =========================
    st.sidebar.header("⚙️ Hyperparameters")
    alpha = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01)
    iterations = st.sidebar.slider("Iterations", 100, 1000, 300)
    lambdas = st.sidebar.multiselect(
        "Lambda values",
        [0, 0.01, 0.1, 1, 10],
        default=[0.01, 0.1, 1]
    )

    # =========================
    # Metrics
    # =========================
    def mse(y, y_hat):
        return np.mean((y - y_hat) ** 2)

    def rmse(y, y_hat):
        return np.sqrt(mse(y, y_hat))

    def r2(y, y_hat):
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot

    # =========================
    # Gradient Descent
    # =========================
    def ridge_gd(X, y, alpha, lam, iterations):
        m, n = X.shape
        theta = np.zeros((n, 1))

        cost_hist, rmse_hist, r2_hist = [], [], []

        for _ in range(iterations):
            y_hat = X @ theta
            error = y_hat - y

            reg = lam * np.r_[np.zeros((1,1)), theta[1:]]
            grad = (1/m) * (X.T @ error + reg)

            theta -= alpha * grad

            cost = (1/(2*m)) * (np.sum(error**2) + lam*np.sum(theta[1:]**2))
            cost_hist.append(cost)
            rmse_hist.append(rmse(y, y_hat))
            r2_hist.append(r2(y, y_hat))

        return theta, cost_hist, rmse_hist, r2_hist

    # =========================
    # Lambda Tuning
    # =========================
    best_lambda = None
    best_rmse = float("inf")

    for lam in lambdas:
        theta_tmp, _, _, _ = ridge_gd(X_train, y_train, alpha, lam, iterations)
        val_pred = X_val @ theta_tmp
        current_rmse = rmse(y_val, val_pred)

        if current_rmse < best_rmse:
            best_rmse = current_rmse
            best_lambda = lam

    st.sidebar.success(f"Best Lambda: {best_lambda}")

    # =========================
    # Train Final Model
    # =========================
    theta, cost_hist, rmse_hist, r2_hist = ridge_gd(
        X_train, y_train, alpha, best_lambda, iterations
    )

    # Predictions
    train_pred = X_train @ theta
    val_pred = X_val @ theta
    test_pred = X_test @ theta

    # =========================
    # Metrics Table
    # =========================
    metrics_df = pd.DataFrame({
        "Dataset": ["Train", "Validation", "Test"],
        "R2": [
            r2(y_train, train_pred),
            r2(y_val, val_pred),
            r2(y_test, test_pred)
        ],
        "RMSE": [
            rmse(y_train, train_pred),
            rmse(y_val, val_pred),
            rmse(y_test, test_pred)
        ],
        "MSE": [
            mse(y_train, train_pred),
            mse(y_val, val_pred),
            mse(y_test, test_pred)
        ]
    })

    st.subheader("📊 Model Performance")
    st.dataframe(metrics_df)

    # =========================
    # Plots
    # =========================
    st.subheader("📈 Training Curves")

    fig1, ax1 = plt.subplots()
    ax1.plot(cost_hist)
    ax1.set_title("Cost vs Iterations")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(rmse_hist)
    ax2.set_title("RMSE vs Iterations")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    ax3.plot(r2_hist)
    ax3.set_title("R2 vs Iterations")
    st.pyplot(fig3)
