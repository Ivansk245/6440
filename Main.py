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
    "patient data via the MIMIC-IV dataset to model the results. Below is a quick comparison between the two. We hope this tool " \
    "guides users in the right direction in terms of their treatment, but users should keep in mind that "
    "<span style='color:red; font-weight:bold;'>this is not medical advice.</span>",
    unsafe_allow_html=True
)

st.write("")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Synthea")
    st.markdown("**Pros:**")
    st.markdown("""
    - Easily Accesbible   
    - Generalized diagnoses 
    - Dont have to worry about privacy
    """)

with col2:
    st.markdown("###")
    st.markdown("**Cons:**")
    st.markdown("""
    - Not real patient data  
    - May miss rare or complex conditions  
    - Unbalanced data  
    """)

with col1:
    st.markdown("### MIMIC-IV")
    st.markdown("**Pros:**")
    st.markdown("""
    - Real ICU patient data
    - Specific data
    - Used for clinical research and modeling
    """)

with col2:
    st.markdown("###")
    st.markdown("**Cons:**")
    st.markdown("""
    - Requires access approval
    - May contain missing or inconsistent data
    - There are numerous medications associated with a single diagnosis 
    """)

st.write("###")

logo1 = os.path.join(BASE_DIR,"pngs", "gatech.png")
logo2 = os.path.join(BASE_DIR,"pngs", "synthea.png")
logo3 = os.path.join(BASE_DIR,"pngs", "physionet.png")
logo4 = os.path.join(BASE_DIR,"pngs", "drugs_com.png")

col1, col2 = st.columns(2)
with col1:
    st.image(logo1, width=250)
with col2:
    st.image(logo2, width=250)

st.write("")

col3, col4 = st.columns(2)
with col3:
    st.image(logo3, width=250)
with col4:
    st.image(logo4, width=250)

