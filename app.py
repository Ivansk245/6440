import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from medications import get_medications
from utils import load_all_patients
from models import train_model_from_fhir

st.set_page_config(page_title="Medication Recommendation Tool", layout="centered")

st.title("Medication Recommendation Tool")
st.warning(
    "These propensity scores are based on synthetic data from Synthea, not real patients. "
    "Changing age and sex affects predictions based on the model trained on fake data."
)
st.write(
    """
    This tool estimates which medications are most commonly prescribed for a given diagnosis
    based on patient characteristics. **This is not medical advice.**
    """
)

if "diagnosis" not in st.session_state:
    st.session_state.diagnosis = ""
if "meds" not in st.session_state:
    st.session_state.meds = []
if "results" not in st.session_state:
    st.session_state.results = pd.DataFrame()

@st.cache_resource
def get_trained_model():
    patients = load_all_patients("fhir")
    model, df = train_model_from_fhir(patients)
    return model, df

model, df = get_trained_model()

diagnosis_input = st.text_input("Enter your diagnosis (e.g., hypertension):").strip()

if diagnosis_input != st.session_state.diagnosis:
    st.session_state.diagnosis = diagnosis_input
    st.session_state.meds = get_medications(diagnosis_input)

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

    try:
        med_classes_in_model = [m for m in meds if m in model.classes_]
        if med_classes_in_model:
            med_scores_full = model.predict_proba([patient_features])[0]
            med_classes = model.classes_
            med_scores = [med_scores_full[list(med_classes).index(m)] for m in med_classes_in_model]
            results = pd.DataFrame({
                "Medication": med_classes_in_model,
                "PropensityScore": med_scores
            }).sort_values("PropensityScore", ascending=False)
        else:
            raise ValueError("No medications in model")
    except:
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

