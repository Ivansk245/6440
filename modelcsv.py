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

    sex_map = {
    "male": 0,
    "m": 0,
    "female": 1,
    "f": 1
    }

    sex_str = str(patient_row.get("GENDER", "other")).lower()
    sex = sex_map.get(sex_str, 0.5)

    #n_conditions = patient_row.get("num_conditions", 0)
    #return [age, sex, n_conditions]
    return [age, sex]

#def train_models_from_csv(patients_csv="patients.csv", medications_csv="medications.csv"):
def train_models_from_csv(patients_csv, medications_csv, diagnosis_count):
    patients_df = pd.read_csv(patients_csv)
    meds_df = pd.read_csv(medications_csv)
    meds_df = meds_df.drop_duplicates(subset=["PATIENT", "DESCRIPTION", "REASONDESCRIPTION"])

    tier_models = {}
    meds_seen = set()

    merged_df = meds_df.merge(patients_df, on="PATIENT", how="left")
    merged_df = merged_df.drop_duplicates(subset=["PATIENT", "DESCRIPTION", "REASONDESCRIPTION"])

    for diagnosis, group in merged_df.groupby("REASONDESCRIPTION"):
        meds_counts = group["DESCRIPTION"].value_counts()
        meds_seen.update(meds_counts.index.tolist())

        if len(meds_counts) == 1:
            tier_models[diagnosis] = {"type": "single", "med": meds_counts.index[0]}
        elif meds_counts.max() < diagnosis_count:
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

def find_matching_diagnoses(user_input, tier_models):
    user_input = user_input.lower().strip()
    matches = [diag for diag in tier_models.keys() if user_input in diag]
    return matches

def predict_combined_propensity(user_input, patient_features, tier_models):
    matching_diags = find_matching_diagnoses(user_input, tier_models)
    if not matching_diags:
        return None

    all_results = []

    for diag in matching_diags:
        tier = tier_models[diag]

        if tier["type"] == "single":
            all_results.append({"Medication": tier["med"], "PropensityScore": 1.0})
        elif tier["type"] == "frequency":
            counts = tier["counts"]
            total = sum(counts.values())
            for med, count in counts.items():
                all_results.append({"Medication": med, "PropensityScore": count / total})
        else:  
            model = tier["model"]
            meds_in_model = model.classes_
            probs = model.predict_proba([patient_features])[0]
            for med, prob in zip(meds_in_model, probs):
                all_results.append({"Medication": med, "PropensityScore": prob})

    df = pd.DataFrame(all_results)
    df = df.groupby("Medication", as_index=False).sum()  
    df["PropensityScore"] = df["PropensityScore"] / df["PropensityScore"].sum()  
    return df.sort_values("PropensityScore", ascending=False)
