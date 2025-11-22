import streamlit as st

st.set_page_config(page_title="Propensity Score Calculator", layout="centered")
st.title("Propensity Score Calculator")
st.warning(
    "These propensity scores are based on real data from the MIMIC-IV Dataset.This " \
    "dataset dataset contains detailed records of ICU patients, capturing more specific diagnoses "
)

st.markdown(
    "**MIMIC-IV** from Physionet is not open-source database. It contains real de-idenfitied patient data. " \
    "that provides researchers with rich clinical data which allows them to study disease progression, treatment outcomes, "
    "and develop predictive models in healthcare."
)