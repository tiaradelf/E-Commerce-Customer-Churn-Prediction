import streamlit as st
from PIL import Image

def tampilkan_home():

    st.header("📌 Business Background")
    st.write("""
    Dalam industri e-commerce yang sangat kompetitif, memahami perilaku pelanggan menjadi hal yang krusial untuk mempertahankan pelanggan yang sudah ada.
    Salah satu tantangan terbesar adalah *customer churn*, yaitu pelanggan yang berhenti menggunakan layanan. 
    Dengan menggunakan analisis data dan teknik machine learning, kita dapat mengidentifikasi faktor-faktor yang menyebabkan churn dan memprediksi pelanggan yang berisiko tinggi untuk churn.
    
    Project ini bertujuan untuk:
    - Menganalisis karakteristik pelanggan berdasarkan data historis.
    - Mengidentifikasi fitur-fitur penting yang memengaruhi churn.
    - Membangun model prediksi churn untuk mendukung keputusan bisnis yang lebih tepat sasaran.
    """)