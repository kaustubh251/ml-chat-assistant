#pip install streamlit pandas numpy matplotlib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Chat ML Assistant", layout="wide")
st.title("💬 Chat-Driven ML Assistant (Ridge Regression)")

# ===============================
# Session Memory (RAG-lite)
# ===============================
if "model_state" not in st.session_state:
    st.session_state.model_state = {}

if "chat" not in st.session_state:
    st.session_state.chat = []

# ===============================
# File Upload
# ===============================
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# ===============================
# Utility Functions
# ===============================
def mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)

def rmse(y, y_hat):
    return np.sqrt(mse(y, y_hat))

def r2(y, y_hat):
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot

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

# ===============================
# Load & Prepare Data
# ===============================
if uploaded_file:
    df = pd.read_csv(uploaded_file).dropna()

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)

    X = (X - X.mean(axis=0)) / X.max(axis=0)
    X = np.c_[np.ones(X.shape[0]), X]

    m = X.shape[0]
    idx = np.random.permutation(m)
    t1, t2 = int(0.6*m), int(0.8*m)

    X_train, X_val, X_test = X[idx[:t1]], X[idx[t1:t2]], X[idx[t2:]]
    y_train, y_val, y_test = y[idx[:t1]], y[idx[t1:t2]], y[idx[t2:]]

    st.success("Dataset loaded and preprocessed")

# ===============================
# Chat Input
# ===============================
prompt = st.chat_input("Ask me to train, evaluate, or explain the model")

if prompt:
    st.session_state.chat.append(("user", prompt))

    response = ""

    # 🧠 Intent Routing (Agent Logic)
    if "train" in prompt.lower():
        best_rmse = float("inf")
        best_lambda = None

        for lam in [0, 0.01, 0.1, 1]:
            theta, _, _, _ = ridge_gd(X_train, y_train, 0.01, lam, 300)
            val_pred = X_val @ theta
            score = rmse(y_val, val_pred)
            if score < best_rmse:
                best_rmse = score
                best_lambda = lam

        theta, cost_hist, rmse_hist, r2_hist = ridge_gd(
            X_train, y_train, 0.01, best_lambda, 300
        )

        st.session_state.model_state = {
            "theta": theta,
            "cost": cost_hist,
            "rmse": rmse_hist,
            "r2": r2_hist,
            "lambda": best_lambda
        }

        response = f"✅ Model trained successfully with λ = {best_lambda}"

    elif "metric" in prompt.lower() or "performance" in prompt.lower():
        theta = st.session_state.model_state["theta"]
        response = f"""
Train R²: {r2(y_train, X_train @ theta):.3f}  
Validation R²: {r2(y_val, X_val @ theta):.3f}  
Test R²: {r2(y_test, X_test @ theta):.3f}
"""

    elif "plot" in prompt.lower():
        fig, ax = plt.subplots()
        ax.plot(st.session_state.model_state["rmse"])
        ax.set_title("RMSE vs Iterations")
        st.pyplot(fig)
        response = "📈 RMSE plot displayed."

    elif "lambda" in prompt.lower():
        response = (
            "λ controls regularization strength. Higher λ reduces variance "
            "but increases bias by shrinking coefficients."
        )

    else:
        response = (
            "I can train the model, show metrics, plot curves, "
            "or explain concepts like lambda and overfitting."
        )

    st.session_state.chat.append(("assistant", response))

# ===============================
# Display Chat
# ===============================
for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.write(msg)
