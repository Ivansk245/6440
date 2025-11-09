import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def extract_features_from_patient_row(patient_row):
    birthDate = patient_row.get("BIRTHDATE", "2000-01-01")
    
    try:
        birth_date_parsed = pd.to_datetime(birthDate)
        birth_year = birth_date_parsed.year
    except Exception:
        birth_year = 2000  
    
    age = pd.Timestamp.now().year - birth_year

    sex_map = {"male": 0, "female": 1}
    sex_str = str(patient_row.get("GENDER", "other")).lower()
    sex = sex_map.get(sex_str, 0.5)

    n_conditions = patient_row.get("num_conditions", 0)
    return [age, sex, n_conditions]

def train_model_from_csv(patients_csv="patients.csv", medications_csv="medications.csv"):
    patients_df = pd.read_csv(patients_csv)
    meds_df = pd.read_csv(medications_csv)

    X = []
    y = []
    meds_seen = set()

    merged_df = meds_df.merge(patients_df, on="PATIENT", how="left")

    for _, row in merged_df.iterrows():
        features = extract_features_from_patient_row(row)
        X.append(features)

        med_name = row["DESCRIPTION"] 
        y.append(med_name)
        meds_seen.add(med_name)

    X = np.array(X)
    y = np.array(y)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    df = pd.DataFrame({
        "Id": merged_df["PATIENT"],
        "medication": y,
        "features": list(X)
    })

    return model, df, list(meds_seen)
