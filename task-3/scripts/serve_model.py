from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

# Initialize FastAPI app
app = FastAPI(title="Rossmann Sales Prediction API")

# Load the serialized model
MODEL_DIR = "../task-2/models/"
MODEL_NAME = None

# Find the latest model (based on timestamp)
for file in os.listdir(MODEL_DIR):
    if file.endswith(".pkl"):
        MODEL_NAME = file

if not MODEL_NAME:
    raise FileNotFoundError("No serialized model found in the models directory.")

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
model = joblib.load(MODEL_PATH)

# Define request body structure
class PredictionRequest(BaseModel):
    Store: int
    DayOfWeek: int
    Promo: int
    StateHoliday: str
    SchoolHoliday: int
    CompetitionDistance: float
    StoreType: str
    Assortment: str
    Year: int
    Month: int
    Day: int
    IsWeekend: int

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Rossmann Sales Prediction API"}

# Prediction endpoint
@app.post("/predict/")
def predict(request: PredictionRequest):
    try:
        # Convert request data to DataFrame
        input_data = pd.DataFrame([request.dict()])
        
        # Preprocess the input data if necessary (the model pipeline includes preprocessing)
        prediction = model.predict(input_data)
        
        return {
            "Store": request.Store,
            "Predicted_Sales": float(prediction[0])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
