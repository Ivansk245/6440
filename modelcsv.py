import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

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

def train_models_from_csv(patients_csv="patients.csv", medications_csv="medications.csv"):
    patients_df = pd.read_csv(patients_csv)
    meds_df = pd.read_csv(medications_csv)

    merged_df = meds_df.merge(patients_df, on="PATIENT", how="left")
    diagnosis_counts = merged_df['REASONDESCRIPTION'].value_counts()

    tier_models = {} 
    meds_seen = set(merged_df['DESCRIPTION'].unique())

    for diagnosis, count in diagnosis_counts.items():
        subset = merged_df[merged_df['REASONDESCRIPTION'] == diagnosis]
        
        X = []
        y = []
        for _, row in subset.iterrows():
            features = extract_features_from_patient_row(row)
            X.append(features)
            y.append(row['DESCRIPTION'])

        if len(X) == 0:
            continue

        X = np.array(X)
        y = np.array(y)

        if count > 1000:
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
        elif 100 <= count <= 1000:
            model = DecisionTreeClassifier()
            model.fit(X, y)
        else:
            model = None  

        tier_models[diagnosis] = {
            "count": count,
            "model": model,
            "X": X,
            "y": y
        }

    return tier_models, merged_df, list(meds_seen)

def predict_propensity(diagnosis, patient_features, tier_models):
    info = tier_models.get(diagnosis)
    if info is None:
        return None  

    count = info['count']
    model = info['model']

    if model is not None:
        probs = model.predict_proba([patient_features])[0]
        meds = model.classes_
        return pd.DataFrame({
            "Medication": meds,
            "PropensityScore": probs
        }).sort_values("PropensityScore", ascending=False)
    else:
        unique_meds = np.unique(info['y'])
        if count == 1:
            return pd.DataFrame({
                "Medication": unique_meds,
                "PropensityScore": [1.0]
            })
        else:
            counts = pd.Series(info['y']).value_counts()
            probs = counts / counts.sum()
            return pd.DataFrame({
                "Medication": probs.index,
                "PropensityScore": probs.values
            }).sort_values("PropensityScore", ascending=False)