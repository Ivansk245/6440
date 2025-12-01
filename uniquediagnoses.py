

"""meds_df1 = pd.read_csv("csv/medications.csv")
meds_df2 = pd.read_csv("csv/mimic_medications.csv")

diagnosis_counts1 = meds_df1['REASONDESCRIPTION'].dropna().value_counts()
diagnosis_df1 = diagnosis_counts1.reset_index()
diagnosis_df1.columns = ['REASONDESCRIPTION', 'COUNT']
diagnosis_df1.to_csv("unique_diagnoses_with_counts1.csv", index=False)

diagnosis_counts2 = meds_df2['REASONDESCRIPTION'].dropna().value_counts()
diagnosis_df2 = diagnosis_counts2.reset_index()
diagnosis_df2.columns = ['REASONDESCRIPTION', 'COUNT']
diagnosis_df2.to_csv("unique_diagnoses_with_counts2.csv", index=False)"""

import pandas as pd

meds_df2 = pd.read_csv("csv/mimic_medications.csv")

diag = "Unspecified essential hypertension"
diag_meds = meds_df2[meds_df2['REASONDESCRIPTION'] == diag]

med_counts = diag_meds['DESCRIPTION'].value_counts().reset_index()
med_counts.columns = ['Medication', 'COUNT']

med_counts = med_counts.sort_values('COUNT', ascending=False)
med_counts.to_csv("tobacco_med_counts.csv", index=False)
