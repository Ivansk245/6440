import streamlit as st
import os
from PIL import Image

BASE_DIR = os.path.dirname(__file__)

st.set_page_config(page_title="Propensity Score Calculator", layout="centered")
st.title("Propensity Score Calculator")

st.write("")

st.markdown(
    "**Propensity Score** is the likelihood of a certain medication prescribed to an individual based on their characteristics. "
    "This tool aims to estimate these propensity scores when an individual inputs some of their demographics like age and sex. "
    "This tool currently utilizes 2 datatsets. Fictional but realistic patient data courtesy of Synthea® dataset and deintified real " \
    "patient data via the MIMIC-IV dataset to model the results. We hope this tool " \
    "guides users in the right direction in terms of their treatment, but users should keep in mind that "
    "<span style='color:red; font-weight:bold;'>this is not medical advice.</span>",
    unsafe_allow_html=True
)

st.write("")

logo1 = os.path.join(BASE_DIR, "gatech.png")
logo2 = os.path.join(BASE_DIR, "synthea.png")
logo3 = os.path.join(BASE_DIR, "physionet.png")

col1, col2, col3 = st.columns(3)

with col1:
    st.image(logo1, width=200)
with col2:
    st.image(logo2, width=200)
with col3:
    st.image(logo3, width=200)
