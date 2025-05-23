import streamlit as st

import streamlit as st
st.set_page_config(page_title="E-commerce Customer Churn Prediction", layout="wide", page_icon=":rocket:")

st.title("Streamlytics: E-commerce Customer Churn Prediction")

st.header("📊 E-Commerce Customer Churn Prediction")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Home", "About Me", "Eksplorasi Data (EDA)", "Prediction", "Business Impact Insight", "Contact"])

if page == "Home":
    import home
    home.tampilkan_home()
elif page == "About Me":
    import about_me
    about_me.tampilkan_about_me()
elif page == "Contact":
    import contact
    contact.tampilkan_contact()
elif page == "Prediction":
    import prediction
    prediction.tampilkan_prediction()
elif page == "Eksplorasi Data (EDA)":
    import eda
    eda.tampilkan_eda()
elif page == "Business Impact Insight":
    import business_insight
    business_insight.tampilkan_business_insight()