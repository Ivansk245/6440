import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def extract_features_from_patient(patient_resource):
    from datetime import datetime
    birthDate = patient_resource.get("birthDate", "2000-01-01")
    birth_year = int(birthDate[:4])
    age = datetime.now().year - birth_year

    sex_map = {"male": 0, "female": 1}
    sex_str = patient_resource.get("gender", "other").lower()
    sex = sex_map.get(sex_str, 0.5)

    n_conditions = 0  
    return [age, sex, n_conditions]

def train_model_from_fhir(patients):
    X = []
    y = []

    for patient_bundle in patients:
        patient_res = next((e for e in patient_bundle.get("entry", []) if e.get("resource", {}).get("resourceType") == "Patient"), None)
        if not patient_res:
            continue

        features = extract_features_from_patient({"resource": patient_res.get("resource", {})})

        for entry in patient_bundle.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "MedicationRequest":
                med_name = resource.get("medicationCodeableConcept", {}).get("text", "UnknownMed")
                X.append(features)
                y.append(med_name)

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0 or len(y) == 0:
        raise ValueError("No valid training data found in FHIR patients.")

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model, pd.DataFrame({"medication": y, "features": list(X)})

