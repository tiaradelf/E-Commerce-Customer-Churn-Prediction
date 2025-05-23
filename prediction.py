import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns

def tampilkan_prediction():
    st.title("📊 E-Commerce Customer Churn Prediction")

    file_path = "E Commerce Dataset.xlsx"
    if not os.path.exists(file_path):
        st.error(f"File '{file_path}' tidak ditemukan di direktori lokal.")
        return

    try:
        df = pd.read_excel(file_path, sheet_name="E Comm")
    except Exception as e:
        st.error(f"Gagal membaca sheet 'E Comm': {e}")
        return

    st.subheader("Data Preview (Sheet: E Comm)")
    st.dataframe(df.head())

    if 'Churn' not in df.columns:
        st.error("Kolom 'Churn' tidak ditemukan di sheet 'E Comm'.")
        return

    if df['Churn'].dtype == 'object':
        mapping = {val: idx for idx, val in enumerate(sorted(df['Churn'].unique()))}
        df['Churn'] = df['Churn'].map(mapping)
        st.write(f"Kolom Churn sudah di-mapping ke numerik: {mapping}")

    if 'CustomerID' in df.columns:
        df.drop(columns='CustomerID', inplace=True)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()

    df[num_cols] = SimpleImputer(strategy='median').fit_transform(df[num_cols])
    df[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[cat_cols])

    encoder = OneHotEncoder(drop='first', sparse_output=False)
    df_encoded = pd.DataFrame(encoder.fit_transform(df[cat_cols]), columns=encoder.get_feature_names_out(cat_cols))

    df_final = pd.concat([df[num_cols], df_encoded], axis=1)

    X = df_final.drop(columns='Churn', errors='ignore')
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_train_probs = model.predict_proba(X_train)[:, 1]
    y_test_pred = model.predict(X_test)
    y_test_probs = model.predict_proba(X_test)[:, 1]

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_precision = precision_score(y_train, y_train_pred)
    test_precision = precision_score(y_test, y_test_pred)
    train_recall = recall_score(y_train, y_train_pred)
    test_recall = recall_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    train_auc = roc_auc_score(y_train, y_train_probs)
    test_auc = roc_auc_score(y_test, y_test_probs)

    st.subheader("Confusion Matrix - Train & Test")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_train, y_train_pred), display_labels=model.classes_).plot(ax=axs[0], cmap="Blues")
    axs[0].set_title("Train")
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_pred), display_labels=model.classes_).plot(ax=axs[1], cmap="Oranges")
    axs[1].set_title("Test")
    st.pyplot(fig)

    st.subheader("ROC Curve - Train vs Test")
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_probs)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_probs)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(fpr_train, tpr_train, label=f'Train AUC = {train_auc:.2f}', linestyle='--')
    ax2.plot(fpr_test, tpr_test, label=f'Test AUC = {test_auc:.2f}', color='orange')
    ax2.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    st.subheader("Evaluasi Model: Train vs Test")
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    train_scores = [train_accuracy, train_precision, train_recall, train_f1]
    test_scores = [test_accuracy, test_precision, test_recall, test_f1]
    x = np.arange(len(metrics))
    width = 0.35
    fig3, ax3 = plt.subplots()
    ax3.bar(x - width/2, train_scores, width, label='Train', color='skyblue')
    ax3.bar(x + width/2, test_scores, width, label='Test', color='orange')
    ax3.set_ylabel("Score")
    ax3.set_title("Train vs Test Evaluation")
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.set_ylim(0, 1.1)
    ax3.legend()
    ax3.grid(axis='y', linestyle='--', alpha=0.6)
    st.pyplot(fig3)

    st.subheader("🔍 Top 10 Fitur yang Paling Mempengaruhi Churn")
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_[0]
    }).sort_values(by='Coefficient', key=abs, ascending=False).head(10)
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.barplot(data=coef_df, x='Coefficient', y='Feature', palette='coolwarm', ax=ax4)
    ax4.set_title('Top 10 Feature Influencers')
    st.pyplot(fig4)

    st.subheader("📌 Model Insights")
    st.markdown(f"""
    - **Train vs Test ROC AUC**: {train_auc:.2f} vs {test_auc:.2f}. menunjukkan bahwa model generalisasi dengan baik dan dapat diandalkan untuk prediksi churn dalam konteks bisnis.
    - **Precision & Recall** penting untuk meminimalkan salah prediksi churn.
    - **Confusion Matrix** menunjukkan model bisa memisahkan churn vs non-churn dengan baik.
    - **Top Fitur** yang mempengaruhi churn dapat dijadikan dasar untuk intervensi bisnis.
    - Jika fitur seperti `Late Deliveries` atau `Customer Complaints` memiliki koefisien tinggi, pertimbangkan perbaikan layanan di area tersebut.
    """)
