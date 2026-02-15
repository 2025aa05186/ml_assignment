import streamlit as st
import pandas as pd
import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)

import matplotlib.pyplot as plt
import seaborn as sns


# ======================================================
# PAGE
# ======================================================
st.set_page_config(page_title="ML Classification App", layout="wide")
st.title("Machine Learning Classification & Evaluation")


# ======================================================
# LOAD FILES
# ======================================================
@st.cache_resource
def load_files():
    models = {
        "Logistic Regression": joblib.load("logistic_regression.pkl"),
        "Decision Tree": joblib.load("decision_tree.pkl"),
        "KNN": joblib.load("knn.pkl"),
        "Naive Bayes": joblib.load("naive_bayes.pkl"),
        "Random Forest": joblib.load("random_forest.pkl"),
        "XGBoost": joblib.load("xgboost.pkl"),
    }

    scaler = joblib.load("scaler.pkl")
    imputer = joblib.load("imputer.pkl")
    encoders = joblib.load("encoders.pkl")
    target_encoder = joblib.load("target_encoder.pkl")
    feature_columns = joblib.load("feature_columns.pkl")

    return models, scaler, imputer, encoders, target_encoder, feature_columns


models, scaler, imputer, encoders, target_encoder, feature_columns = load_files()


# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.header("Controls")

model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
model = models[model_name]

uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type=["csv"])


# ======================================================
# PREDICTION
# ======================================================
if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.write(data.head())

    # assume last column is target
    X = data.iloc[:, :-1].copy()
    y = data.iloc[:, -1].copy()

    # -----------------------------------------
    # DEFENSIVE PREPROCESSING
    # -----------------------------------------

    # remove spaces
    X.columns = X.columns.str.strip()

    # add missing columns if any
    for col in feature_columns:
        if col not in X.columns:
            X[col] = np.nan

    # keep only required
    X = X[feature_columns]

    # encode categorical
    for col, encoder in encoders.items():
        try:
            X[col] = encoder.transform(X[col].astype(str))
        except:
            X[col] = np.nan

    # convert numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # ⭐ CRITICAL FOR CLOUD
    X = X.values

    # impute
    X = imputer.transform(X)

    # scale
    X = scaler.transform(X)

    # encode target
    y = target_encoder.transform(y)

    # -----------------------------------------
    # PREDICT
    # -----------------------------------------
    y_pred = model.predict(X)

    # ======================================================
    # METRICS
    # ======================================================
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y, y_pred)

    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)
            auc = roc_auc_score(y, y_prob, multi_class="ovr")
        else:
            auc = "N/A"
    except:
        auc = "N/A"

    st.subheader("Evaluation Metrics")

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{acc:.4f}")
    c1.metric("Precision", f"{prec:.4f}")
    c2.metric("Recall", f"{rec:.4f}")
    c2.metric("F1 Score", f"{f1:.4f}")
    c3.metric("MCC", f"{mcc:.4f}")
    c3.metric("AUC", auc if isinstance(auc, str) else f"{auc:.4f}")

    # ======================================================
    # CONFUSION MATRIX
    # ======================================================
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

else:
    st.info("Upload a CSV file from sidebar.")


# ======================================================
# FOOTER
# ======================================================
st.write("---")
st.write("Developed for ML Assignment")