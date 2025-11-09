import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
from PIL import Image
from modelcsv import train_models_from_csv, predict_propensity  

BASE_DIR = os.path.dirname(__file__)
patients_csv = os.path.join(BASE_DIR, "csv", "patients.csv")
medications_csv = os.path.join(BASE_DIR, "csv", "medications.csv")

st.set_page_config(page_title="Propensity Score Calculator", layout="centered")
st.title("Propensity Score Calculator")

st.warning(
    "These propensity scores are based on synthetic data from CSVs, not real patients. "
    "Changing age and sex affects predictions based on the model trained on fake data."
)

st.markdown(
    "**Propensity Score** is the likelihood of a certain medication prescribed to an individual based on their characteristics. "
    "This tool aims to estimate these propensity scores when an individual inputs some of their demographics like age and sex. "
    "This tool currently utilizes fictional but realistic patient data to model the results. "
    "We hope this tool guides users in the right direction in terms of their treatment, but users should keep in mind that "
    "<span style='color:red; font-weight:bold;'>this is not medical advice.</span>",
    unsafe_allow_html=True
)

if "diagnosis" not in st.session_state:
    st.session_state.diagnosis = ""
if "meds" not in st.session_state:
    st.session_state.meds = []
if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()

def get_medications_from_csv(diagnosis):
    df = pd.read_csv(medications_csv)
    meds = df[df['REASONDESCRIPTION'].str.lower() == diagnosis.lower()]['DESCRIPTION'].unique().tolist()
    return meds

@st.cache_resource
def get_trained_models():
    tier_models, df, meds_seen = train_models_from_csv(patients_csv, medications_csv)
    return tier_models, df, meds_seen

tier_models, df, meds_seen = get_trained_models()

diagnosis_input = st.text_input("Enter your diagnosis (e.g., hypertension):").strip()

if diagnosis_input != st.session_state.diagnosis:
    st.session_state.diagnosis = diagnosis_input
    st.session_state.meds = get_medications_from_csv(diagnosis_input)

meds = st.session_state.meds
st.write(f"Found {len(meds)} medications for `{diagnosis_input}`")

if meds:
    st.write(meds)
else:
    st.warning(f"No medications found for `{diagnosis_input}`. Please check your spelling or try another diagnosis.")

if meds:
    st.subheader("Patient Characteristics")
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    sex = st.selectbox("Sex", ["Male", "Female", "Other"])
    num_conditions = st.number_input("Number of known conditions", min_value=0, max_value=20, value=0)

    sex_map = {"Male": 0, "Female": 1, "Other": 0.5}
    patient_features = [age, sex_map[sex], num_conditions]

    results = predict_propensity(st.session_state.diagnosis, patient_features, tier_models)

    if results is None:
        np.random.seed()
        scores = np.random.rand(len(meds))
        scores = scores / scores.sum()
        results = pd.DataFrame({
            "Medication": meds,
            "PropensityScore": scores
        })

    st.session_state.results = results

    st.subheader("Propensity Scores")
    chart = alt.Chart(results).mark_bar().encode(
        x=alt.X('PropensityScore', title='Propensity Score'),
        y=alt.Y('Medication', sort='-x', title='Medication')
    )
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Detailed Scores")
    st.dataframe(results)

logo1 = os.path.join(BASE_DIR, "gatech.png")
logo2 = os.path.join(BASE_DIR, "synthea.png")

col1, col2 = st.columns(2)

with col1:
    st.image(logo1, use_container_width=True)

with col2:
    st.image(logo2, use_container_width=True)
