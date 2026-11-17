import pandas as pd
import numpy as np
import ast
from src.utils.helpers import load_object

_cache = {}

def _load_data():
    if "data" not in _cache:
        _cache["data"] = {
            "model": load_object("models/Random-Forest-Model.joblib"),
            "le": load_object("models/Label-Encoder.joblib"),
            "training_df": pd.read_csv("data/processed/train/Training.csv", nrows=1),
            "diet_df": pd.read_csv("data/processed/diet/diets.csv"),
            "medications_df": pd.read_csv("data/processed/medication/medications.csv"),
            "description_df": pd.read_csv("data/processed/description/description.csv"),
            "precautions_df": pd.read_csv("data/processed/precaution/precautions_df.csv", index_col=[0]),
            "symptoms_df": pd.read_csv("data/processed/symptoms/symtoms_df.csv", index_col=[0]),
            "symptoms_severity_df": pd.read_csv("data/processed/symptom-severity/Symptom-severity.csv"),
            "workout_df": pd.read_csv("data/processed/workout/workout_df.csv", index_col=[0, 1]),
        }
        _cache["data"]["symptom_columns"] = _cache["data"]["training_df"].drop(columns=["prognosis"]).columns
    return _cache["data"]


def predict(symptoms: str) -> dict:
    d = _load_data()

    symptoms_list = [s.strip().lower().replace(" ", "_") for s in symptoms.split(",")]
    symptoms_list = [s.replace("__", "_") for s in symptoms_list]
    input_vector = np.zeros(len(d["symptom_columns"]))
    for symptom in symptoms_list:
        index = d["symptom_columns"].get_loc(symptom)
        input_vector[index] = 1

    input_df = pd.DataFrame([input_vector], columns=d["symptom_columns"])
    prediction = d["model"].predict(input_df)[0]
    disease = d["le"].classes_[prediction]

    severity_score = d["symptoms_severity_df"][
        d["symptoms_severity_df"]["Symptom"].isin(symptoms_list)
    ]["weight"].sum()

    diets = d["diet_df"][d["diet_df"]["Disease"].str.strip().str.lower() == disease.strip().lower()]["Diet"].values[0]
    workouts = d["workout_df"][d["workout_df"]["disease"].str.strip().str.lower() == disease.strip().lower()]["workout"].tolist()
    precautions = d["precautions_df"][d["precautions_df"]["Disease"].str.strip().str.lower() == disease.strip().lower()][
        ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]].values[0]
    medications = d["medications_df"][d["medications_df"]["Disease"].str.strip().str.lower() == disease.strip().lower()]["Medication"].values[0]
    description = d["description_df"][d["description_df"]["Disease"].str.strip().str.lower() == disease.strip().lower()]["Description"].values[0]

    medications = ast.literal_eval(medications)
    diets = ast.literal_eval(diets)
    precautions = [p for p in precautions if pd.notna(p)]

    return {
        "disease": disease,
        "description": description,
        "severity_score": int(severity_score),
        "precautions": precautions,
        "medications": medications,
        "diets": diets,
        "workouts": workouts
    }   