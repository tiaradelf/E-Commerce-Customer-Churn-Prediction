# streamlit_churn_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
)

st.set_page_config(page_title="E-commerce Customer Churn Prediction", layout="wide", page_icon=":rocket:")

@st.cache_data
def load_data(path="E Commerce Dataset.xlsx"):
    return pd.read_excel(path, sheet_name="E Comm")

@st.cache_data
def preprocess_data(df):
    if 'CustomerID' in df.columns:
        df.drop(columns='CustomerID', inplace=True)
    if df['Churn'].dtype == 'object':
        df['Churn'] = df['Churn'].map({val: idx for idx, val in enumerate(sorted(df['Churn'].unique()))})

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    df[num_cols] = SimpleImputer(strategy='median').fit_transform(df[num_cols])
    df[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[cat_cols])

    encoder = OneHotEncoder(drop='first', sparse_output=False)
    df_encoded = pd.DataFrame(encoder.fit_transform(df[cat_cols]), columns=encoder.get_feature_names_out(cat_cols))

    df_final = pd.concat([df[num_cols], df_encoded], axis=1)
    X = df_final.drop(columns='Churn', errors='ignore')
    y = df['Churn']

    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42), X.columns

@st.cache_resource
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def plot_evaluation(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_probs = model.predict_proba(X_train)[:, 1]
    y_test_probs = model.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_train, y_train_probs)
    test_auc = roc_auc_score(y_test, y_test_probs)

    st.subheader("Confusion Matrix - Train & Test")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_train, y_train_pred)).plot(ax=axs[0], cmap="Blues")
    axs[0].set_title("Train")
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_pred)).plot(ax=axs[1], cmap="Oranges")
    axs[1].set_title("Test")
    st.pyplot(fig)

    st.subheader("ROC Curve - Train vs Test")
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_probs)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_probs)
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr_train, tpr_train, label=f'Train AUC = {train_auc:.2f}', linestyle='--')
    ax2.plot(fpr_test, tpr_test, label=f'Test AUC = {test_auc:.2f}', color='orange')
    ax2.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax2.legend()
    ax2.set_title("ROC Curve")
    st.pyplot(fig2)

    return train_auc, test_auc, y_train_pred, y_test_pred

def tampilkan_prediction():
    st.title("📊 E-Commerce Customer Churn Prediction")
    df = load_data()
    (X_train, X_test, y_train, y_test), feature_names = preprocess_data(df)
    model = train_model(X_train, y_train)
    
    train_auc, test_auc, y_train_pred, y_test_pred = plot_evaluation(model, X_train, X_test, y_train, y_test)

    st.subheader("📌 Model Insights")
    st.markdown(f"""
    - **Train vs Test ROC AUC**: {train_auc:.2f} vs {test_auc:.2f}.
    - **Confusion Matrix** menunjukkan performa klasifikasi yang stabil.
    - Insight lebih lanjut dapat dilihat dari fitur-fitur penting.
    """)

    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0]
    }).sort_values(by='Coefficient', key=abs, ascending=False).head(10)

    st.subheader("Top 10 Fitur yang Paling Mempengaruhi Churn")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.barplot(data=coef_df, x='Coefficient', y='Feature', palette='coolwarm', ax=ax3)
    ax3.set_title('Top Feature Influencers')
    st.pyplot(fig3)

if __name__ == '__main__':
    tampilkan_prediction()