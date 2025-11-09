import pandas as pd

# Load medications CSV
meds_df = pd.read_csv("csv/medications.csv")

# Count occurrences of each non-empty diagnosis
diagnosis_counts = meds_df['REASONDESCRIPTION'].dropna().value_counts()

# Convert to DataFrame
diagnosis_df = diagnosis_counts.reset_index()
diagnosis_df.columns = ['REASONDESCRIPTION', 'COUNT']

# Save to a new CSV file
diagnosis_df.to_csv("unique_diagnoses_with_counts.csv", index=False)

print("Unique diagnoses with counts saved to unique_diagnoses_with_counts.csv")