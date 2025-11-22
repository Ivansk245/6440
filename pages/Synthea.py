import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
from PIL import Image
from modelcsv import train_models_from_csv, predict_propensity

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
patients_csv = os.path.join(BASE_DIR, "csv", "patients.csv")
medications_csv = os.path.join(BASE_DIR, "csv", "medications.csv")

st.set_page_config(page_title="Propensity Score Calculator", layout="centered")
st.title("Propensity Score Calculator")
st.warning(
    "These propensity scores are based on synthetic data from the Synthea Dataset. This " \
    "datset covers more generalized diagnose's' but is limited to the amount"
)

st.markdown(
    "**Synthea** is an open-source, synthetic patient generator that models the medical history of synthetic patients. " \
    " The resulting data is free from cost, privacy, and security " \
    "restrictions, enabling research with Health IT data that is otherwise legally or practically unavailable.",
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

diagnosis_input = st.text_input("Enter your diagnosis (e.g., hypertension):").strip().lower()

if diagnosis_input != st.session_state.diagnosis:
    st.session_state.diagnosis = diagnosis_input
    st.session_state.meds = get_medications_from_csv(diagnosis_input)

meds = st.session_state.meds
#st.write(f"Found {len(meds)} medications for `{diagnosis_input}`")
if meds:
    #st.write(meds)
    st.write(f"Found {len(meds)} medications for `{diagnosis_input}`")
else:
    st.warning(f"No medications found for `{diagnosis_input}`. Please check your spelling or try another diagnosis.")

if meds:
    st.subheader("Patient Characteristics")
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    sex = st.selectbox("Sex", ["Male", "Female"])
    #num_conditions = st.number_input("Number of known conditions", min_value=0, max_value=20, value=0)

    sex_map = {"Male": 0, "Female": 1}
    #patient_features = [age, sex_map[sex], num_conditions]
    patient_features = [age, sex_map[sex]]

    results = predict_propensity(diagnosis_input, patient_features, tier_models)
    if results is not None:
        st.session_state.results = results
    else:
        st.warning("No model available for this diagnosis.")
    
    if "selected_chart" not in st.session_state:
        st.session_state.selected_chart = "Bar Chart" 

    st.subheader("Propensity Scores")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Bar Chart"):
            st.session_state.selected_chart = "Bar Chart"
    with col2:
        if st.button("Stacked Bar Chart"):
            st.session_state.selected_chart = "Stacked Bar Chart"
    with col3:
        if st.button("Pie Chart"):
            st.session_state.selected_chart = "Pie Chart"

    df_results = st.session_state.results

    bar_chart = alt.Chart(df_results).mark_bar().encode(
        x='PropensityScore',
        y=alt.Y('Medication', sort='-x'),
        color=alt.Color('PropensityScore', scale=alt.Scale(scheme='blues'))
    )

    stacked_bar = alt.Chart(df_results).mark_bar().encode(
        x=alt.X('Patient:N', title='Patient'),
        y=alt.Y('PropensityScore:Q', title='Propensity Score'),
        color=alt.Color('Medication:N', title='Medication'),
        tooltip=['Medication', 'PropensityScore']
    ).properties(width=600,height=400)

    pie_chart = alt.Chart(df_results).mark_arc(innerRadius=0).encode(
        theta='PropensityScore',
        color='Medication'
    )

    if st.session_state.selected_chart == "Bar Chart":
        st.altair_chart(bar_chart, use_container_width=True)
    elif st.session_state.selected_chart == "Stacked Bar Chart":
        st.altair_chart(stacked_bar, use_container_width=True)
    elif st.session_state.selected_chart == "Pie Chart":
        st.altair_chart(pie_chart, use_container_width=True)

    st.subheader("Detailed Scores")
    st.dataframe(df_results)


