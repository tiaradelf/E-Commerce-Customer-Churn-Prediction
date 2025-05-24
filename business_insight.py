import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_resource
def tampilkan_business_insight():

    st.title("📌 Business Impact Insight")

    st.header("🔍 Business Impact Insight")

    st.subheader("📊 Korelasi dengan variabel target Churn")
    st.markdown("""
    - **Tenure** memiliki korelasi negatif paling kuat dengan churn (**-0.34**). Semakin lama pelanggan bertahan, semakin kecil kemungkinan mereka churn.
    - **CashbackAmount** juga berkorelasi negatif (**-0.15**). Pelanggan yang menerima lebih banyak cashback cenderung lebih loyal.
    - Fitur lain seperti `SatisfactionScore`, `OrderCount`, `CouponUsed`, dan `NumberOfAddress` hanya memiliki korelasi **lemah** (di bawah 0.1) terhadap churn.
    - Korelasi lemah menunjukkan pengaruh kecil, namun tetap bisa berperan sebagai pendukung model prediktif.
    """)

    st.subheader("✅ Insight Strategis")
    st.markdown("""
    - **Customer loyalty** sangat dipengaruhi oleh lamanya waktu menjadi pelanggan (**Tenure**) dan nilai cashback yang diterima.
    - Pelanggan yang tinggal lebih jauh dari warehouse cenderung churn → bisa diatasi dengan **strategi logistik** yang lebih baik (misalnya: ekspansi gudang).
    - Fitur dengan pengaruh kecil (korelasi rendah) sebaiknya dipertimbangkan untuk **dieliminasi dari model** agar lebih efisien.
    - Beberapa hasil seperti `SatisfactionScore` yang **lebih tinggi di churner** bisa menunjukkan adanya **data bias atau label noise**, dan perlu ditinjau ulang kualitas datanya.
    """)

    st.subheader("📍 Recommendation (Rekomendasi Strategis)")
    st.markdown("""
    ### 🎯 1. Tingkatkan Retensi Pelanggan Baru
    - Fokus pada pelanggan baru dengan **Tenure rendah**.
    - Terapkan **insentif onboarding**: cashback awal, kupon diskon berjenjang, atau produk yang dipersonalisasi.

    ### 💰 2. Optimalkan Program Cashback
    - Evaluasi kembali kebijakan cashback untuk pelanggan dengan risiko churn tinggi.
    - Buat **sistem cashback berjenjang** berdasarkan loyalitas pelanggan.

    ### 🚚 3. Perbaiki Distribusi Logistik
    - Pelanggan jauh dari warehouse lebih mungkin churn.
    - Ekspansi lokasi gudang atau kerja sama logistik lokal untuk **meningkatkan kecepatan pengiriman**.

    ### 🧠 4. Manfaatkan Model untuk Deteksi Dini
    - Gunakan output model untuk **intervensi proaktif** terhadap pelanggan yang diprediksi akan churn.
    - Hubungkan prediksi dengan sistem CRM/Marketing Automation.

    ### 🧹 5. Lakukan Feature Pruning untuk Efisiensi
    - Fitur dengan korelasi lemah bisa dihapus untuk **menyederhanakan model**.
    - Efisiensi sangat penting saat model diterapkan ke lingkungan produksi.

    ### 🔍 6. Tinjau Kualitas Data dan Label
    - Beberapa anomali bisa disebabkan oleh **label noise atau kesalahan input data**.
    - Audit data dapat membantu **meningkatkan akurasi model**.
    """)