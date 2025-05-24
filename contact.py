import streamlit as st
from PIL import Image
import os

@st.cache_resource
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

    st.markdown(
        """
        <div style='text-align: center; font-size: 18px; margin-top: 20px;'>
            🌟 Terima kasih telah mengeksplorasi Project Data Science ini! 🌟<br>
            Semoga hasil analisis dan insight yang diberikan dapat bermanfaat dalam pengambilan keputusan bisnis yang lebih baik.
        </div>
        """, unsafe_allow_html=True
    )

    st.divider()

    st.header("🎨 Till Next Time")

    image_path = "ucapan.jpg"  # Pastikan file ini ada di folder utama Streamlit-mu

    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption="Terima kasih telah menjelajahi streamlit ini!", use_container_width=True)
    else:
        st.warning("❗ Gambar 'ucapan.jpg' tidak ditemukan. Pastikan file berada di direktori yang sama.")