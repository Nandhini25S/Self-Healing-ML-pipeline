"""
main.py - API Endpoints with FastAPI
Run this and go to http://127.0.0.1:5000/docs to test
"""

import json
import logging
import uvicorn
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from stage0 import load_model_artifacts, make_prediction, CONFIG


app = FastAPI(title="Self-Healing ML Pipeline API",
              description="Stage 0: Bank Marketing Prediction API", version="1.0.0")


loaded_model = None
loaded_preprocessing = None
model_version = "v1"


class PredictionInput(BaseModel):
    """Input schema for prediction endpoint"""
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    emp_var_rate: float = Field(alias="emp.var.rate")
    cons_price_idx: float = Field(alias="cons.price.idx")
    cons_conf_idx: float = Field(alias="cons.conf.idx")
    euribor3m: float
    nr_employed: float = Field(alias="nr.employed")

    class Config:
        populate_by_name = True
        schema_extra = {
            "example": {"age": 30, "job": "admin.", "marital": "married", "education": "university.degree",
                        "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
                        "month": "may", "day_of_week": "mon", "campaign": 1, "pdays": 999, "previous": 0,
                        "poutcome": "nonexistent", "emp.var.rate": 1.1, "cons.price.idx": 93.994,
                        "cons.conf.idx": -36.4, "euribor3m": 4.857, "nr.employed": 5191.0}}


class PredictionOutput(BaseModel):
    """Output schema for prediction endpoint"""
    prediction: int
    prediction_label: str
    prediction_proba: dict
    confidence: float
    timestamp: str
    model_version: str

class HealthResponse(BaseModel):
    """Output schema for health endpoint"""
    status: str
    model_version: str
    timestamp: str



def setup_logging():
    """Configure logging for inference."""
    Path(CONFIG["logs_dir"]).mkdir(exist_ok=True)
    logging.basicConfig(filename=f"{CONFIG['logs_dir']}/inference.log", level=logging.INFO, format='%(message)s')


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global loaded_model, loaded_preprocessing

    print("="*60)
    print("🚀 Starting API Server...")
    print("="*60)

    try:
        print("\nLoading model artifacts...")
        loaded_model, loaded_preprocessing = load_model_artifacts()

        setup_logging()
        print("✓ Logging configured")

        print("\n" + "="*60)
        print("✅ SERVER READY")
        print("="*60)
        print("\n📡 API Documentation: http://127.0.0.1:5000/docs")
        print("📝 Logs: logs/inference.log")
        print("="*60 + "\n")

    except FileNotFoundError:
        print("\n❌ ERROR: Model not found!")
        print("   Run: python stage0.py")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.
    Returns server status and model version.
    """
    return {"status": "healthy", "model_version": model_version, "timestamp": datetime.now().isoformat()}

@app.post("/predict", response_model=PredictionOutput)
async def predict(data: PredictionInput):
    """
    Make a prediction.
    Provide customer information and get subscription prediction.
    """
    try:
        timestamp = datetime.now().isoformat()

        # Convert to dict with proper field names for the model
        input_data = {"age": data.age, "job": data.job, "marital": data.marital, "education": data.education,
                      "default": data.default, "housing": data.housing, "loan": data.loan, "contact": data.contact,
                      "month": data.month, "day_of_week": data.day_of_week, "campaign": data.campaign,
                      "pdays": data.pdays, "previous": data.previous, "poutcome": data.poutcome,
                      "emp.var.rate": data.emp_var_rate, "cons.price.idx": data.cons_price_idx,
                      "cons.conf.idx": data.cons_conf_idx, "euribor3m": data.euribor3m, "nr.employed": data.nr_employed}

        # Make prediction
        result = make_prediction(loaded_model, loaded_preprocessing, input_data)

        # Add metadata
        result["timestamp"] = timestamp
        result["model_version"] = model_version

        # Log the inference
        log_entry = {"timestamp": timestamp, "input_features": input_data, "prediction": result["prediction"],
                     "prediction_proba": result["prediction_proba"]["yes"], "model_version": model_version}
        logging.info(json.dumps(log_entry))

        # Remove processed_features from response
        result.pop("processed_features", None)
        return result

    except Exception as e:
        logging.error(json.dumps({"timestamp": datetime.now().isoformat(), "error": str(e), "input_data": str(data)}))
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")