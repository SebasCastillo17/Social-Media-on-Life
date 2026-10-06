from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = Path(__file__).resolve().parents[1] / "machine_learning" / "random_forest_gpa.joblib"

app = FastAPI(title="Social Media Impact - GPA Predictor")
model = joblib.load(MODEL_PATH)


class StudentFeatures(BaseModel):
    Age: int
    Daily_Usage_Hours: float
    Weekend_Extra_Hours: float
    Sleep_Quality_Score: int
    Perceived_Stress_Score: float
    Mental_Health_Index: int
    Gender: str
    Academic_Level: str
    Primary_Platform: str
    Device_Type: str
    Late_Night_Usage: bool
    Social_Comparison_Frequency: str


@app.get("/")
def root():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(features: StudentFeatures):
    # The pipeline's ColumnTransformer selects columns by name, so order doesn't matter.
    df = pd.DataFrame([features.model_dump()])
    predicted_gpa = model.predict(df)[0]
    return {"predicted_gpa": round(float(predicted_gpa), 2)}
