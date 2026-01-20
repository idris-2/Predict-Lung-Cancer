from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

# Import the wrapper so joblib knows how to reconstruct the TensorFlow model
from keras_classifier import KerasBinaryClassifier

app = FastAPI(title="Lung Cancer Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict later in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the folder where models are stored
MODEL_DIR = "Models"

MODEL_PATHS = {
    "XGBoost": f"{MODEL_DIR}/XGBoost.pkl",
    "TensorFlow": f"{MODEL_DIR}/TensorFlow (Keras).pkl",
    "SVM": f"{MODEL_DIR}/SVM (RBF).pkl"
}

# Explicitly define column order to match training data
# If your training script dropped "lung_cancer", this must be that exact column list
COLUMN_ORDER = [
    "gender", "age", "smoking", "yellow_fingers", "anxiety", "peer_pressure",
    "chronic_disease", "fatigue", "severe_fatigue", "allergy", "wheezing",
    "alcohol", "coughing", "shortness_of_breath",
    "swallowing_difficulty", "chest_pain", "environmental_risk",
    "symptom_count", "respiratory_score", "systemic_score",
    "severe_respiratory", "age_risk"
]

loaded_models = {}

# Load models on startup
@app.on_event("startup")
def load_all_models():
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            try:
                loaded_models[name] = joblib.load(path)
                print(f"Successfully loaded {name}")
            except Exception as e:
                print(f"Error loading {name}: {e}")
        else:
            print(f"Warning: Model file not found at {path}")

class PatientInput(BaseModel):
    gender: int
    age: int
    smoking: int
    yellow_fingers: int
    anxiety: int
    peer_pressure: int
    chronic_disease: int
    fatigue: int
    severe_fatigue: int
    allergy: int
    wheezing: int
    alcohol: int
    coughing: int
    shortness_of_breath: int
    swallowing_difficulty: int
    chest_pain: int
    environmental_risk: int

@app.post("/predict")
def predict(data: PatientInput):
    if not loaded_models:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # 1. Feature Engineering (must match training script)
    symptom_count = sum([
        data.smoking,
        data.yellow_fingers,
        data.anxiety,
        data.peer_pressure,
        data.chronic_disease,
        data.fatigue,
        data.severe_fatigue,
        data.allergy,
        data.wheezing,
        data.alcohol,
        data.coughing,
        data.shortness_of_breath,
        data.swallowing_difficulty,
        data.chest_pain,
        data.environmental_risk
    ])

    respiratory_score = (
        data.coughing + data.wheezing + data.shortness_of_breath + data.chest_pain
    )
    severe_respiratory = int(data.shortness_of_breath == 1 and data.chest_pain == 1)
    systemic_score = (
        data.fatigue +
        data.severe_fatigue +
        data.anxiety +
        data.chronic_disease +
        data.allergy
    )
    age_risk = 1 if data.age >= 60 else 0
    
    # 2. Prepare Input DataFrame with strict column ordering
    input_data = data.model_dump()
    input_data.update({
        "symptom_count": symptom_count,
        "systemic_score": systemic_score,
        "respiratory_score": respiratory_score,
        "severe_respiratory": severe_respiratory,
        "age_risk": age_risk
    })
    
    # Force order to match training
    X = pd.DataFrame([input_data])[COLUMN_ORDER]

    # 3. Ensemble Prediction Logic
    try:
        # Get individual probabilities
        results_map = {}
        for name, model in loaded_models.items():
            results_map[name] = float(model.predict_proba(X)[0][1])

        all_probs = list(results_map.values())
        
        # Majority voting strategy (threshold 0.7)
        high_votes = [p for p in all_probs if p >= 0.7]

        if len(high_votes) >= (len(all_probs) // 2 + 1):
            final_prob = np.mean(high_votes)
        else:
            final_prob = np.mean(all_probs)

        # Risk level categorization
        risk_level = "high" if final_prob >= 0.7 else "medium" if final_prob >= 0.4 else "low"

        return {
            "lung_cancer_probability": round(float(final_prob), 4),
            "risk_level": risk_level,
            "individual_model_probs": {name: round(p, 4) for name, p in results_map.items()}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")