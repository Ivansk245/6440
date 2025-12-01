import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
from modelcsv import train_models_from_csv, predict_propensity, predict_combined_propensity

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
patients_csv = os.path.join(BASE_DIR, "csv", "mimic_patients.csv")
medications_csv = os.path.join(BASE_DIR, "csv", "mimic_medications.csv")

st.set_page_config(page_title="Propensity Score Calculator", layout="centered")
st.title("Propensity Score Calculator")
st.warning(
    "These propensity scores are based on real data from the MIMIC-IV Dataset. This " \
    "dataset contains detailed records of ICU patients, capturing more specific diagnoses."
)

st.markdown(
    "**MIMIC-IV** from Physionet is not an open-source database. It contains real de-identified patient data that provides researchers with rich clinical data which allows them to study disease progression, treatment outcomes, and develop predictive models in healthcare."
)

page_prefix = "MIMIC"
diag_key = f"{page_prefix}_diagnosis"
meds_key = f"{page_prefix}_meds"
results_key = f"{page_prefix}_results"
chart_key = f"{page_prefix}_selected_chart"

if diag_key not in st.session_state:
    st.session_state[diag_key] = ""
if meds_key not in st.session_state:
    st.session_state[meds_key] = []
if results_key not in st.session_state:
    st.session_state[results_key] = pd.DataFrame()
if chart_key not in st.session_state:
    st.session_state[chart_key] = "Bar Chart"

def get_medications_from_csv(diagnosis_input):
    df = pd.read_csv(medications_csv)
    meds = df[df['REASONDESCRIPTION'].str.lower().str.contains(diagnosis_input.lower())]['DESCRIPTION'].unique().tolist()
    return meds

@st.cache_resource
def get_trained_models(patients_csv, medications_csv, modeling_choice):
    if modeling_choice == "Frequency-based Modeling (Recommended)":
        diagnosis_count = 100
    else:
        diagnosis_count = 20
    
    tier_models, df, meds_seen = train_models_from_csv(patients_csv, medications_csv, diagnosis_count)
    return tier_models, df, meds_seen

modeling_choice = st.radio(
    "Choose Modeling Approach",
    ["Frequency-based Modeling (Recommended)", "Logistic Regression"]
)

diagnosis_input = st.text_input("Enter your diagnosis (e.g., hypertension):").strip().lower()

if diagnosis_input != st.session_state[diag_key]:
    st.session_state[diag_key] = diagnosis_input
    st.session_state[meds_key] = get_medications_from_csv(diagnosis_input)

meds = st.session_state[meds_key]

if diagnosis_input:
    if meds:
        st.write(f"Found {len(meds)} medications for `{diagnosis_input}`")
    else:
        st.warning(f"No medications found for `{diagnosis_input}`. Please check your spelling or try another diagnosis.")
        all_diags = pd.read_csv(medications_csv)['REASONDESCRIPTION'].dropna().unique().tolist()
        all_diags = sorted([d for d in all_diags if isinstance(d, str)])

        with st.expander("Here are all available diagnoses:"):
            st.write(all_diags)

if meds:
    st.subheader("Patient Characteristics")
    age = st.number_input("Age", min_value=0, max_value=120, value=0)
    sex = st.selectbox("Sex", ["", "Male", "Female"])

    if age > 0 and sex:
        if st.button("Calculate Propensity Scores"):
            sex_map = {"Male": 0, "Female": 1}
            patient_features = [age, sex_map[sex]]

            tier_models, df, meds_seen = get_trained_models(
                patients_csv, medications_csv, modeling_choice
            )

            results = predict_combined_propensity(diagnosis_input, patient_features, tier_models)
            if results is not None:
                results = results.copy()
                results['PropensityScore'] = results['PropensityScore'].astype(float)
                st.session_state[results_key] = results.sort_values("PropensityScore", ascending=False).head(20)
            else:
                st.warning("No model available for this diagnosis.")
                st.stop()

if st.session_state[results_key] is not None and not st.session_state[results_key].empty:
    df_results = st.session_state[results_key]

    st.subheader("Propensity Scores")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Bar Chart"):
            st.session_state[chart_key] = "Bar Chart"
    with col2:
        if st.button("Pie Chart"):
            st.session_state[chart_key] = "Pie Chart"

    bar_chart = alt.Chart(df_results).mark_bar().encode(
        x='PropensityScore',
        y=alt.Y('Medication', sort='-x'),
        color=alt.Color('PropensityScore', scale=alt.Scale(scheme='blues'))
    )

    pie_chart = alt.Chart(df_results).mark_arc(innerRadius=0).encode(
        theta='PropensityScore',
        color='Medication'
    )

    if st.session_state[chart_key] == "Bar Chart":
        st.altair_chart(bar_chart, use_container_width=True)
    else:
        st.altair_chart(pie_chart, use_container_width=True)

    st.warning(
        "Keep in mind that because these are ICU patients, a diagnosis may yield additional medications "
        "for surgical prep, associated infections, etc."
    )

    df_links = pd.DataFrame({
        "Medication (Drugs.com)": df_results["Medication"].apply(
            lambda med: f'<a href="https://www.drugs.com/search.php?searchterm={med}" target="_blank">{med}</a>'),
        "PropensityScore": df_results["PropensityScore"]})

    st.subheader("Detailed Scores")
    st.markdown(df_links.to_html(escape=False, index=False), unsafe_allow_html=True)

    csv_data = df_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Results as CSV",
        data=csv_data,
        file_name="propensity_scores.csv",
        mime="text/csv"
    )