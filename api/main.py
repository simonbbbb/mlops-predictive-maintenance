from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import json
import pickle
import os
import time
import logging
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import mlflow.pyfunc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_latest_model():
    metadata_dir = 'models/metadata'
    metadata_files = [os.path.join(metadata_dir, f) for f in os.listdir(metadata_dir) 
                       if f.startswith('metadata_') and f.endswith('.json')]
    if not metadata_files:
        raise FileNotFoundError("No model metadata found")
    latest_metadata_file = sorted(metadata_files)[-1]
    with open(latest_metadata_file, 'r') as f:
        metadata = json.load(f)
    model_path = metadata['model_path']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(metadata['feature_names_path'], 'r') as f:
        feature_names = json.load(f)['feature_names']
    return model, feature_names, metadata

app = FastAPI(
    title="Predictive Maintenance API",
    description="API for predicting equipment failures",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    global model, feature_names, model_metadata
    try:
        model, feature_names, model_metadata = load_latest_model()
        logger.info(f"Loaded model: {model_metadata['model_path']}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        model, feature_names, model_metadata = None, None, None

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

class SensorData(BaseModel):
    equipment_id: int
    timestamp: str
    temperature: float
    vibration: float
    pressure: float
    noise_level: float
    class Config:
        schema_extra = {
            "example": {
                "equipment_id": 1,
                "timestamp": "2023-05-01T10:00:00",
                "temperature": 76.2,
                "vibration": 0.54,
                "pressure": 98.6,
                "noise_level": 62.3
            }
        }

class BatchSensorData(BaseModel):
    data: List[SensorData]

class PredictionResponse(BaseModel):
    prediction_id: str
    equipment_id: int
    timestamp: str
    failure_probability: float
    prediction: int
    model_version: str
    prediction_time: str

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    summary: Dict[str, Any]

@app.get("/")
def read_root():
    return {"message": "Predictive Maintenance API", "status": "active"}

@app.get("/health")
def health_check():
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Model not loaded"}
        )
    return {"status": "healthy", "model_info": {"type": model_metadata["model_type"], "version": model_metadata["timestamp"]}}

@app.get("/model/info")
def model_info():
    if model_metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not available")
    return {
        "model_type": model_metadata["model_type"],
        "version": model_metadata["timestamp"],
        "accuracy": model_metadata["accuracy"],
        "f1_score": model_metadata["f1_score"],
        "parameters": model_metadata["parameters"]
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(data: SensorData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        input_data = pd.DataFrame([data.dict()])
        input_data['timestamp'] = pd.to_datetime(input_data['timestamp'])
        input_data['hour'] = input_data['timestamp'].dt.hour
        input_data['day_of_week'] = input_data['timestamp'].dt.dayofweek
        missing_features = set(feature_names) - set(input_data.columns)
        for feature in missing_features:
            input_data[feature] = 0
        input_features = input_data[feature_names]
        prediction_proba = model.predict_proba(input_features)[0][1]
        prediction = 1 if prediction_proba >= 0.5 else 0
        response = PredictionResponse(
            prediction_id=str(uuid.uuid4()),
            equipment_id=data.equipment_id,
            timestamp=data.timestamp,
            failure_probability=float(prediction_proba),
            prediction=int(prediction),
            model_version=model_metadata["timestamp"],
            prediction_time=datetime.now().isoformat()
        )
        logger.info(f"Prediction made: equipment_id={data.equipment_id}, prob={prediction_proba:.4f}, pred={prediction}")
        return response
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(batch_data: BatchSensorData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        predictions = []
        for sensor_data in batch_data.data:
            data_dict = sensor_data.dict()
            input_data = pd.DataFrame([data_dict])
            input_data['timestamp'] = pd.to_datetime(input_data['timestamp'])
            input_data['hour'] = input_data['timestamp'].dt.hour
            input_data['day_of_week'] = input_data['timestamp'].dt.dayofweek
            missing_features = set(feature_names) - set(input_data.columns)
            for feature in missing_features:
                input_data[feature] = 0
            input_features = input_data[feature_names]
            prediction_proba = model.predict_proba(input_features)[0][1]
            prediction = 1 if prediction_proba >= 0.5 else 0
            predictions.append(
                PredictionResponse(
                    prediction_id=str(uuid.uuid4()),
                    equipment_id=sensor_data.equipment_id,
                    timestamp=sensor_data.timestamp,
                    failure_probability=float(prediction_proba),
                    prediction=int(prediction),
                    model_version=model_metadata["timestamp"],
                    prediction_time=datetime.now().isoformat()
                )
            )
        prediction_values = [p.prediction for p in predictions]
        summary = {
            "total_records": len(predictions),
            "predicted_failures": sum(prediction_values),
            "predicted_ok": len(prediction_values) - sum(prediction_values),
            "failure_rate": sum(prediction_values) / len(prediction_values) if prediction_values else 0,
            "batch_id": str(uuid.uuid4()),
            "processed_at": datetime.now().isoformat()
        }
        return BatchPredictionResponse(predictions=predictions, summary=summary)
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 