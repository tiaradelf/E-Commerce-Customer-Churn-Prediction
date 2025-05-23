import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
)

@st.cache_data
def tampilkan_eda():
    st.title("🔍 Eksplorasi Data (EDA) - Customer Churn")

    file_path = "E Commerce Dataset.xlsx"
    if not os.path.exists(file_path):
        st.error(f"File '{file_path}' tidak ditemukan di direktori lokal.")
        return

    try:
        df = pd.read_excel(file_path, sheet_name="E Comm")
    except Exception as e:
        st.error(f"Gagal membaca sheet 'E Comm': {e}")
        return

    # 1. Distribusi Churn
    st.subheader("Distribusi Churn Pelanggan")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='Churn', data=df, palette='Set2', ax=ax1)
    ax1.set_title('Distribusi Churn Pelanggan')
    ax1.set_xlabel('Churn (1 = Ya, 0 = Tidak)')
    ax1.set_ylabel('Jumlah Pelanggan')
    st.pyplot(fig1)
    st.markdown("- Terlihat bahwa jumlah pelanggan yang churn lebih sedikit dari yang tidak churn.")

    # 2. Persentase Churn per Gender
    st.subheader("Persentase Churn berdasarkan Gender")
    gender_grouped = df.groupby('Gender')['Churn'].value_counts(normalize=True).unstack() * 100
    fig2, axes = plt.subplots(1, 2, figsize=(10, 5))
    for i, gender in enumerate(['Female', 'Male']):
        axes[i].pie(gender_grouped.loc[gender],
                    labels=['Tidak Churn', 'Churn'],
                    autopct='%.1f%%',
                    colors=['#a0d8ef', '#ffb6b9'],
                    startangle=90,
                    explode=(0, 0.1))
        axes[i].set_title(gender)
    st.pyplot(fig2)
    st.markdown("- Proporsi churn relatif mirip antara laki-laki dan perempuan.")

    # 3. Churn per Kategori Order
    st.subheader("Churn berdasarkan Kategori Order yang Disukai")
    df['OrderCategoryGrouped'] = df['PreferedOrderCat'].replace({
        'Laptop & Accessory': 'Electronics',
        'Mobile': 'Electronics',
        'Mobile Phone': 'Electronics',
        'Mobile Phone and Accessory': 'Electronics',
        'Fashion': 'Lifestyle',
        'Grocery': 'Daily Needs',
        'Others': 'Others'
    })
    category_churn = df.groupby('OrderCategoryGrouped')['Churn'].value_counts(normalize=True).unstack() * 100
    category_churn = category_churn.rename(columns={0: 'Not Churned (%)', 1: 'Churned (%)'})
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=category_churn.index, y=category_churn['Churned (%)'], palette='Set2', ax=ax3)
    ax3.set_title('Persentase Churn Berdasarkan Kategori Order')
    ax3.set_ylabel('Churned (%)')
    ax3.set_xlabel('Kategori Order')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
    st.pyplot(fig3)
    st.markdown("- Pelanggan yang memesan kategori lifestyle dan daily needs cenderung memiliki churn lebih tinggi.")

    # --- Korelasi
    st.subheader("Korelasi Antar Fitur Termasuk Churn")
    selected_cols = [
        'Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'NumberOfDeviceRegistered',
        'SatisfactionScore', 'NumberOfAddress', 'OrderAmountHikeFromlastYear',
        'CouponUsed', 'OrderCount', 'CashbackAmount', 'Churn'
    ]
    corr_data = df[selected_cols]
    correlation_matrix = corr_data.corr()
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, ax=ax4)
    ax4.set_title("Heatmap Korelasi Antar Fitur Termasuk Churn")
    st.pyplot(fig4)

    # --- Rata-rata berdasarkan Churn
    st.subheader("Rata-rata Fitur berdasarkan Status Churn")
    numeric_features = selected_cols[:-1]
    churn_grouped = df.groupby('Churn')[numeric_features].mean()
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    sns.heatmap(churn_grouped.T, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax5)
    ax5.set_title("Rata-rata Fitur berdasarkan Status Churn")
    ax5.set_xlabel("Churn")
    ax5.set_ylabel("Fitur")
    st.pyplot(fig5)

    # --- Insight Gabungan
    st.subheader("📌 Insight Korelasi & Strategi")
    st.markdown("""
    **Korelasi dengan variabel target Churn:**
    - `Tenure` memiliki korelasi negatif paling kuat dengan churn **(-0.34)** → semakin lama pelanggan bertahan, semakin kecil kemungkinan mereka churn.
    - `CashbackAmount` juga berkorelasi negatif **(-0.15)** → pelanggan yang mendapat cashback lebih besar cenderung loyal.
    - Fitur seperti `SatisfactionScore`, `OrderCount`, `CouponUsed`, dan `NumberOfAddress` hanya memiliki korelasi **lemah (di bawah 0.1)** terhadap churn → pengaruhnya kecil, namun tetap bisa mendukung model.

    **Insight Strategis:**
    - **Customer loyalty** sangat dipengaruhi oleh **lama menjadi pelanggan (Tenure)** dan **nilai cashback**.
    - Pelanggan yang tinggal lebih jauh dari warehouse cenderung churn → **strategi logistik seperti ekspansi gudang** bisa dipertimbangkan.
    - Fitur dengan korelasi rendah sebaiknya dipertimbangkan untuk **dieliminasi** dalam model agar lebih efisien.
    - Perlu perhatian jika terdapat **anomali** seperti skor kepuasan yang tinggi justru muncul pada pelanggan yang churn → kemungkinan adanya **data bias atau label noise**.
    """)