import os
import json
from datetime import datetime
from pathlib import Path  

def load_all_patients(folder_path):
    patients = []
    folder = Path(folder_path)
    for file in folder.glob("*.json"): 
        with open(file, "r", encoding="utf-8") as f:  
            data = json.load(f)
            patients.append(data)
    return patients

def load_fhir_bundle(path):
    with open(path, "r") as f:
        return json.load(f)

def extract_patient_info(bundle):
    patient_resource = None
    conditions = []
    medications = []

    for entry in bundle.get("entry", []):
        res = entry.get("resource", {})
        rtype = res.get("resourceType")

        if rtype == "Patient":
            patient_resource = res
        elif rtype == "Condition":
            code = res.get("code", {}).get("text")
            if code:
                conditions.append(code.lower())
        elif rtype == "MedicationRequest":
            med = res.get("medicationCodeableConcept", {}).get("text")
            if med:
                medications.append(med.lower())

    if not patient_resource:
        return None

    gender = patient_resource.get("gender", "unknown")
    birth_date = patient_resource.get("birthDate")

    if birth_date:
        birth = datetime.strptime(birth_date, "%Y-%m-%d")
        age = (datetime.now() - birth).days // 365
    else:
        age = None

    return {
        "age": age,
        "sex": gender.capitalize(),
        "conditions": conditions,
        "medications": medications,
    }
