import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def extract_features_from_patient_row(patient_row):
    """Extract features [age, sex, num_conditions] from a patient row."""
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

    #n_conditions = patient_row.get("num_conditions", 0)
    #return [age, sex, n_conditions]
    return [age, sex]

def train_models_from_csv(patients_csv="patients.csv", medications_csv="medications.csv"):
    """Train tiered models for each diagnosis."""
    patients_df = pd.read_csv(patients_csv)
    meds_df = pd.read_csv(medications_csv)

    tier_models = {}
    meds_seen = set()

    merged_df = meds_df.merge(patients_df, on="PATIENT", how="left")

    for diagnosis, group in merged_df.groupby("REASONDESCRIPTION"):
        meds_counts = group["DESCRIPTION"].value_counts()
        meds_seen.update(meds_counts.index.tolist())

        if len(meds_counts) == 1:
            tier_models[diagnosis] = {"type": "single", "med": meds_counts.index[0]}
        elif meds_counts.max() < 100:
            tier_models[diagnosis] = {"type": "frequency", "counts": meds_counts.to_dict()}
        else:
            X = np.array([extract_features_from_patient_row(row) for _, row in group.iterrows()])
            y = group["DESCRIPTION"].values
            model = LogisticRegression(max_iter=1000)
            try:
                model.fit(X, y)
                tier_models[diagnosis] = {"type": "model", "model": model}
            except ValueError:
                tier_models[diagnosis] = {"type": "frequency", "counts": meds_counts.to_dict()}
    
    tier_models = {k.strip().lower(): v for k, v in tier_models.items()}

    return tier_models, merged_df, list(meds_seen)

def predict_propensity(diagnosis, patient_features, tier_models):
    """Predict propensity scores for a given diagnosis and patient."""
    if diagnosis not in tier_models:
        return None

    tier = tier_models[diagnosis]

    if tier["type"] == "single":
        return pd.DataFrame({
            "Medication": [tier["med"]],
            "PropensityScore": [1.0]
        })
    elif tier["type"] == "frequency":
        counts = tier["counts"]
        total = sum(counts.values())
        return pd.DataFrame({
            "Medication": list(counts.keys()),
            "PropensityScore": [v / total for v in counts.values()]
        }).sort_values("PropensityScore", ascending=False)
    else:
        model = tier["model"]
        meds_in_model = model.classes_
        probs = model.predict_proba([patient_features])[0]
        return pd.DataFrame({
            "Medication": meds_in_model,
            "PropensityScore": probs
        }).sort_values("PropensityScore", ascending=False)
