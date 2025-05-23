import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_data():
    return pd.read_excel("E Commerce Dataset.xlsx", sheet_name="E Comm")

def tampilkan_eda():
    st.title("🔍 Eksplorasi Data (EDA) - Customer Churn")

    with st.spinner("⏳ Memuat dan memproses data..."):
        try:
            df = load_data()
        except Exception as e:
            st.error(f"Gagal memuat data: {e}")
            return

        # Distribusi Churn
        st.subheader("Distribusi Churn Pelanggan")
        fig1, ax1 = plt.subplots()
        sns.countplot(x='Churn', data=df, palette='Set2', ax=ax1)
        ax1.set_title('Distribusi Churn Pelanggan')
        ax1.set_xlabel('Churn (1 = Ya, 0 = Tidak)')
        ax1.set_ylabel('Jumlah Pelanggan')
        st.pyplot(fig1)
        st.markdown("- Pelanggan churn lebih sedikit dibandingkan yang loyal.")

        # Churn per Gender
        st.subheader("Persentase Churn berdasarkan Gender")
        gender_grouped = df.groupby('Gender')['Churn'].value_counts(normalize=True).unstack() * 100
        fig2, axes = plt.subplots(1, 2, figsize=(10, 5))
        for i, gender in enumerate(['Female', 'Male']):
            axes[i].pie(
                gender_grouped.loc[gender],
                labels=['Tidak Churn', 'Churn'],
                autopct='%.1f%%',
                colors=['#a0d8ef', '#ffb6b9'],
                startangle=90,
                explode=(0, 0.1)
            )
            axes[i].set_title(gender)
        st.pyplot(fig2)

        # Churn per Preferensi Order
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

        # Korelasi
        st.subheader("Heatmap Korelasi Antar Fitur Termasuk Churn")
        selected_cols = [
            'Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'NumberOfDeviceRegistered',
            'SatisfactionScore', 'NumberOfAddress', 'OrderAmountHikeFromlastYear',
            'CouponUsed', 'OrderCount', 'CashbackAmount', 'Churn'
        ]
        corr_data = df[selected_cols]
        correlation_matrix = corr_data.corr()
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, ax=ax4)
        ax4.set_title("Korelasi Antar Fitur Termasuk Churn")
        st.pyplot(fig4)

        # Rata-rata berdasarkan Churn
        st.subheader("Rata-rata Fitur berdasarkan Status Churn")
        churn_grouped = df.groupby('Churn')[selected_cols[:-1]].mean()
        fig5, ax5 = plt.subplots(figsize=(12, 6))
        sns.heatmap(churn_grouped.T, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax5)
        ax5.set_title("Rata-rata Fitur berdasarkan Status Churn")
        ax5.set_xlabel("Churn")
        ax5.set_ylabel("Fitur")
        st.pyplot(fig5)

        # Insight
        st.subheader("📌 Insight Korelasi & Strategi")
        st.markdown("""
        **Korelasi dengan variabel target Churn:**
        - `Tenure` memiliki korelasi negatif paling kuat dengan churn **(-0.34)**.
        - `CashbackAmount` juga negatif **(-0.15)**.
        - Fitur lain punya korelasi lemah tapi tetap relevan.

        **Insight Strategis:**
        - Customer loyalitas dipengaruhi `Tenure` dan `CashbackAmount`.
        - Jarak ke warehouse tinggi → churn tinggi → pertimbangkan ekspansi gudang.
        - Hapus fitur yang tidak signifikan dari model untuk efisiensi.
        - Periksa anomali seperti SatisfactionScore tinggi pada churner.
        """)