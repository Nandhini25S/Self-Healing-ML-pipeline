"""main.py - API Endpoints with FastAPI"""

import json
import time
import shutil
import logging
import uvicorn
import threading
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from stage0 import load_model_artifacts, make_prediction, CONFIG
from stage1 import run_monitoring_check, get_latest_report, get_current_status, MONITORING_CONFIG, DATASET_CONFIG

app = FastAPI(title="Self-Healing ML Pipeline API", description="Stage 0: Bank Marketing Prediction API", version="1.0.0")

loaded_model = None
loaded_preprocessing = None
model_version = "v1"

prediction_counter = 0
prediction_counter_lock = threading.Lock()
dataset_info = {"name": None, "features": [], "target": None, "path": None}
dataset_lock = threading.Lock()


class PredictionInput(BaseModel):
    """Dynamic input schema - accepts any fields"""
    class Config:
        extra = "allow"  # Allow any additional fields

    def dict(self, **kwargs):
        return super().dict(**kwargs)

    class Config:
        populate_by_name = True
        schema_extra = {
            "example": {"age": 30, "job": "admin.", "marital": "married", "education": "university.degree",
                        "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
                        "month": "may", "day_of_week": "mon", "campaign": 1, "pdays": 999,
                        "previous": 0, "poutcome": "nonexistent", "emp.var.rate": 1.1, "cons.price.idx": 93.994,
                        "cons.conf.idx": -36.4, "euribor3m": 4.857, "nr.employed": 5191.0}}

class PredictionOutput(BaseModel):
    """Output schema for prediction endpoint"""
    prediction: int
    prediction_label: str
    prediction_proba: dict
    confidence: float
    timestamp: str
    model_version: str

    class Config:
        extra = "allow"

class HealthResponse(BaseModel):
    """Output schema for health endpoint"""
    status: str
    model_version: str
    timestamp: str

class MonitoringStatusResponse(BaseModel):
    """Output schema for monitoring status"""
    status: str
    total_predictions: int
    last_check_timestamp: str = None
    rolling_window_size: int

class DatasetUpload(BaseModel):
    """Dataset upload info"""
    dataset_name: str

class DatasetInfo(BaseModel):
    """Dataset metadata"""
    name: str
    features: list
    target: str
    n_samples: int
    n_features: int


def background_monitoring_thread():
    """
    Background thread that monitors prediction count.
    Triggers drift detection every N predictions.
    """
    global prediction_counter

    print("🔍 Background monitoring thread started")
    print(f"   Will check drift every {MONITORING_CONFIG['check_every_n_predictions']} predictions")

    last_check_count = 0

    while True:
        time.sleep(10)
        with prediction_counter_lock:
            current_count = prediction_counter

        # Check if we've crossed the threshold
        if current_count - last_check_count >= MONITORING_CONFIG['check_every_n_predictions']:
            print(f"\n🔔 Prediction count reached {current_count}. Triggering drift detection...")
            try:
                run_monitoring_check()
                last_check_count = current_count
            except Exception as e:
                print(f"❌ Error in background monitoring: {e}")

def setup_logging():
    """Configure logging for inference."""
    Path(CONFIG["logs_dir"]).mkdir(exist_ok=True)
    logging.basicConfig(filename=f"{CONFIG['logs_dir']}/inference.log", level=logging.INFO, format='%(message)s')


@app.on_event("startup")
async def startup_event():
    """Load model on startup and start background monitoring."""
    global loaded_model, loaded_preprocessing

    print("="*60)
    print("🚀 Starting API Server...")
    print("="*60)

    try:
        print("\nLoading model artifacts...")
        loaded_model, loaded_preprocessing = load_model_artifacts()

        setup_logging()
        print("✓ Logging configured")

        # Start background monitoring thread
        monitor_thread = threading.Thread(target=background_monitoring_thread, daemon=True)
        monitor_thread.start()
        print("✓ Background monitoring started")

        print("\n" + "="*60)
        print("✅ SERVER READY")
        print("="*60)
        print("\n📡 API Documentation: http://127.0.0.1:5000/docs")
        print("📝 Logs: logs/inference.log")
        print("📊 Monitoring: Every 100 predictions")
        print("="*60 + "\n")

    except FileNotFoundError:
        print("\n❌ ERROR: Model not found!")
        print("   Run: python stage0.py")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


# @app.post("/dataset/upload")
# async def upload_dataset(file_data: dict):
#     """
#     Upload dataset and extract metadata.
#     Expects: {"dataset_name": str, "file_content": base64 or path}
#     """
#     global dataset_info

#     try:
#         dataset_name = file_data.get("dataset_name")
#         file_path = file_data.get("file_path")

#         # Create datasets directory
#         Path(DATASET_CONFIG["dataset_dir"]).mkdir(exist_ok=True)

#         # Determine file extension
#         ext = Path(file_path).suffix
#         save_path = Path(DATASET_CONFIG["dataset_dir"]) / f"{dataset_name}{ext}"

#         # Copy file
#         shutil.copy(file_path, save_path)

#         # Read dataset
#         if ext == ".csv":
#             df = pd.read_csv(save_path)
#         else:
#             df = pd.read_excel(save_path)

#         # Extract metadata
#         features = df.columns[:-1].tolist()
#         target = df.columns[-1]

#         with dataset_lock:
#             dataset_info = {
#                 "name": dataset_name,
#                 "features": features,
#                 "target": target,
#                 "path": str(save_path),
#                 "n_samples": len(df),
#                 "n_features": len(features)
#             }

#         return {
#             "message": "Dataset uploaded successfully",
#             "dataset_info": dataset_info
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/dataset/upload")
async def upload_dataset(file_data: dict):
    """Upload dataset and extract metadata"""
    global dataset_info

    dataset_name = file_data.get("dataset_name")
    file_path = file_data.get("file_path")

    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=400, detail=f"File not found: {file_path}")

    # Create datasets directory
    Path(DATASET_CONFIG["dataset_dir"]).mkdir(exist_ok=True)

    # Determine file extension
    ext = Path(file_path).suffix
    save_path = Path(DATASET_CONFIG["dataset_dir"]) / f"{dataset_name}{ext}"

    # Copy file
    shutil.copy(file_path, save_path)

    # Read dataset
    if ext == ".csv":
        df = pd.read_csv(save_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(save_path)
    else:
        raise HTTPException(status_code=400, detail="Only CSV and Excel files supported")

    # Extract metadata
    features = df.columns[:-1].tolist()
    target = df.columns[-1]

    dataset_info["name"] = dataset_name
    dataset_info["features"] = features
    dataset_info["target"] = target
    dataset_info["path"] = str(save_path)
    dataset_info["n_samples"] = len(df)
    dataset_info["n_features"] = len(features)

    return {"message": "Dataset uploaded successfully", "dataset_info": dataset_info}

@app.get("/dataset/info", response_model=DatasetInfo)
async def get_dataset_info():
    """Get current dataset metadata"""
    if not dataset_info["name"]:
        raise HTTPException(status_code=404, detail="No dataset uploaded")
    return dataset_info

@app.get("/features/schema")
async def get_feature_schema():
    """Get feature names and types for dynamic form generation"""
    if not dataset_info["features"]:
        if not dataset_info.get("path"):
            raise HTTPException(status_code=404, detail="Dataset path not found")
        raise HTTPException(status_code=404, detail="No dataset uploaded")

    # Read dataset to infer types
    try:
        if dataset_info["path"].endswith(".csv"):
            df = pd.read_csv(dataset_info["path"])
        elif dataset_info["path"].endswith((".xlsx", ".xls")):
            df = pd.read_excel(dataset_info["path"])
        else:
            raise ValueError("Unsupported file format")

        schema = []
        for col in dataset_info["features"]:
            dtype = str(df[col].dtype)
            if "int" in dtype or "float" in dtype:
                field_type = "number"
            else:
                field_type = "text"

            schema.append({
                "name": col,
                "type": field_type,
                "sample_value": str(df[col].iloc[0]) if len(df) > 0 else ""
            })

        return {"features": schema, "target": dataset_info["target"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint. Returns server status and model version"""
    return {"status": "healthy", "model_version": model_version, "timestamp": datetime.now().isoformat()}

@app.post("/predict")
async def predict(data: dict):
    """Make a prediction. Provide customer information and get subscription prediction"""
    global prediction_counter
    try:
        timestamp = datetime.now().isoformat()

        # Use the data directly - no hardcoded field mapping
        input_data = data

        # Make prediction
        result = make_prediction(loaded_model, loaded_preprocessing, input_data)

        # Add metadata
        result["timestamp"] = timestamp
        result["model_version"] = model_version

        # Log the inference
        log_entry = {
            "timestamp": timestamp,
            "input_features": input_data,
            "prediction": result["prediction"],
            "prediction_proba": result["prediction_proba"].get("yes", result["prediction_proba"].get(1, list(result["prediction_proba"].values())[0])),
            "model_version": model_version
        }
        logging.info(json.dumps(log_entry))

        # Increment prediction counter
        with prediction_counter_lock:
            prediction_counter += 1

        # Remove processed_features from response
        result.pop("processed_features", None)

        return result

    except Exception as e:
        logging.error(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "input_data": str(data)
        }))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/status", response_model=MonitoringStatusResponse)
async def monitoring_status():
    """Get current monitoring status. Returns quick overview of drift detection state"""
    try:
        status_info = get_current_status()
        return status_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/latest-report")
async def monitoring_latest_report():
    """Get the latest drift detection report. Returns full drift analysis with all metrics"""
    try:
        report = get_latest_report()
        if not report:
            raise HTTPException(status_code=404, detail="No drift report found. Make predictions and wait for automatic check, or trigger manually.")
        return report
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/monitoring/trigger")
async def monitoring_trigger():
    """Manually trigger drift detection. Forces an immediate drift check regardless of prediction count"""
    try:
        print("\n🔔 Manual drift check triggered via API")
        report = run_monitoring_check()
        if not report:
            raise HTTPException(status_code=400, detail="Could not run drift check. Ensure there are enough predictions logged.")
        return {"message": "Drift check completed", "report": report}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")