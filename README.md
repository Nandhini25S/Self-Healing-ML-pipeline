# Self Healing ML pipeline
It's a self-healing ML pipeline



┌─────────────────────────────────────────────────────────────┐
│                    python stage0.py                         │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Load Data                                          │
│  • Reads bank-additional.csv (semicolon-separated)          │
│  • Drops 'duration' (not available before call happens)     │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: Preprocess                                         │
│  • LabelEncoder for categorical columns (job, marital, etc) │
│  • StandardScaler for numerical columns (age, campaign)     │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: Split Data                                         │
│  • 70% Training → 15% Validation → 15% Test/Baseline        │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: Train Model                                        │
│  • Logistic Regression with balanced class weights          │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 5: Evaluate & Save                                    │
│  • Computes accuracy, precision, recall, F1, ROC-AUC        │
│  • Saves model, preprocessor, and baseline metrics          │
└─────────────────────────────────────────────────────────────┘

Train a model → Save everything → Serve predictions via API → Log for monitoring.


💡 Key Design Choices
1. Drops duration – because in real life, you don't know call duration before the call happens
2. Uses class_weight='balanced' – handles imbalanced data (most people say "no")
3. Saves baseline stats – for future drift detection (this is the "self-healing" foundation)











# Stage 0: Self-Healing ML Pipeline
## 📁 Files

```
stage0.py    # Train model (pure ML logic)
main.py      # API endpoints (FastAPI)
```

---

## 🚀 Setup & Run
### 1. Install
```bash
pip install pandas numpy scikit-learn fastapi uvicorn joblib pydantic
```

### 2. Train Model
```bash
python stage0.py
```

### 3. Start API
```bash
python main.py
```

### 4. Test API
**Go to:** `http://127.0.0.1:5000/docs`

---

## 🧪 What to Test in /docs

### **GET /health**
Click "Try it out" → "Execute"

**Expected response:**
```json
{
  "status": "healthy",
  "model_version": "v1",
  "timestamp": "2024-12-18T10:30:00"
}
```

---

### **GET /info**
Click "Try it out" → "Execute"

**Expected response:**
```json
{
  "model_version": "v1",
  "created_at": "20241218_103000",
  "validation_metrics": {
    "accuracy": 0.899,
    "roc_auc": 0.865,
    "precision": 0.66,
    "recall": 0.47,
    "f1_score": 0.55
  },
  "n_features": 19,
  "feature_names": ["age", "job", "marital", ...]
}
```

---

### **POST /predict**
Click "Try it out" → Use this example input:

```json
{
  "age": 30,
  "job": "admin.",
  "marital": "married",
  "education": "university.degree",
  "default": "no",
  "housing": "yes",
  "loan": "no",
  "contact": "cellular",
  "month": "may",
  "day_of_week": "mon",
  "campaign": 1,
  "pdays": 999,
  "previous": 0,
  "poutcome": "nonexistent",
  "emp_var_rate": 1.1,
  "cons_price_idx": 93.994,
  "cons_conf_idx": -36.4,
  "euribor3m": 4.857,
  "nr_employed": 5191.0
}
```

**Expected response:**
```json
{
  "prediction": 0,
  "prediction_label": "no",
  "prediction_proba": {
    "no": 0.8766,
    "yes": 0.1234
  },
  "confidence": 0.8766,
  "timestamp": "2024-12-18T10:30:00",
  "model_version": "v1"
}
```

**What it means:**
- `prediction: 0` = Customer will NOT subscribe
- `prediction_label: "no"` = Same thing, human-readable
- `confidence: 0.8766` = Model is 87.66% confident
- `prediction_proba.yes: 0.1234` = 12.34% chance of subscription

---

## ✅ Stage 0 Complete When:
- [ ] `python stage0.py` creates models/ and baseline/
- [ ] `python main.py` starts without errors
- [ ] `/docs` page opens at http://127.0.0.1:5000/docs
- [ ] All 3 endpoints work in the interactive docs
- [ ] Predictions are logged to `logs/inference.log`





## Stage 0: establishes a stable production control case.
It trains a baseline logistic regression model on a fixed dataset, defines what “healthy behavior” looks like using static validation metrics and reference baseline statistics, saves all model and preprocessing artifacts, deploys the model behind an inference API, and logs every prediction for future monitoring.


