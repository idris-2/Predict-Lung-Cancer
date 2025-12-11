from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# 1. FastAPI Backend (for Docker)
# Load trained model (update path)
model = joblib.load("lung_model.pkl")

app = FastAPI()

class Symptoms(BaseModel):
    symptom1: int
    symptom2: int
    symptom3: int
    # Add all symptoms your model uses

@app.post("/predict")
def predict(symptoms: Symptoms):
    data = [[
        symptoms.symptom1,
        symptoms.symptom2,
        symptoms.symptom3,
        # ...
    ]]
    
    prediction = model.predict_proba(data)[0][1]  # probability of lung cancer
    return {"probability": float(prediction)}

# ⚠️ Important: CORS on Backend
# Frontend → Backend request requires CORS enabled.
# Add this to main.py:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


"""
POST /predict
{
  "symptom_1": 1,
  "symptom_2": 0,
  ...
}

Returns:

{
  "probability": 0.82
}
"""
"""
Frontend can be:
GitHub Pages
Netlify
Vercel
Hugging Face (but usually frontend stays separate)
"""
"""
🐳 3. Dockerfile (Simple + Correct for Hugging Face)
Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
"""

"""
🚀 5. How to deploy on Hugging Face Spaces (Docker Space)
A Hugging Face Docker Space requires:
Dockerfile
main.py
requirements.txt
lung_model.pkl

No extra config required.
Upload → Build starts automatically → API goes live.
"""