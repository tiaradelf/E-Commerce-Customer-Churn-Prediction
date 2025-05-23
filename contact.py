import streamlit as st
from PIL import Image

def tampilkan_contact():
    st.title("Contact")
    st.write("Contact me through the following link:")

    # LinkedIn
    st.markdown(
        "[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue)](https://www.linkedin.com/in/tiara-delfira/)"
    )

    # GitHub
    st.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-Profile-black)](https://github.com/tiaradelf)"
    )

    # Email
    st.write("📧 Email: delfiratiara7@gmail.com")

    st.divider()

    st.markdown("""
    <div style='text-align: center; font-size: 20px; margin-top: 20px;'>
        🌟 Terima kasih telah mengeksplorasi Project Data Science ini! 🌟<br>
        Semoga hasil analisis dan insight yang diberikan dapat bermanfaat dalam pengambilan keputusan bisnis yang lebih baik.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.header("🎨 Data Scientist")
    # Tampilkan gambar (gunakan use_container_width)
    try:
        image = Image.open("ucapan.jpg")  # Ganti dengan nama file kamu
        st.image(image, caption="Terima kasih telah menjelajahi streamlit ini!", use_container_width=True)
    except FileNotFoundError:
        st.warning("Gambar tidak ditemukan. Pastikan file 'ucapan.png' berada di folder yang sama dengan file Streamlit.")