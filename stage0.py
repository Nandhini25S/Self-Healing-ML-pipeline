# Main pipeline script

"""
Stage 0: Self-Healing ML Pipeline - Problem Setup & Control Case

This script:
1. Loads and splits the bank-additional.csv dataset
2. Trains a logistic regression baseline model
3. Evaluates and saves baseline metrics
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# CONFIGURATION
CONFIG = {"data_path": "bank+marketing/bank-additional/bank-additional.csv",
          "test_size": 0.15, "validation_size": 0.15, "random_state": 42,
          "model_dir": "models", "logs_dir": "logs", "baseline_dir": "baseline"}


# DATA LOADING AND PREPROCESSING
def load_and_preprocess_data(data_path):
    """Load the bank marketing dataset and preprocess it"""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, sep=';')

    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['y'].value_counts()}")

    # Separate features and target
    X = df.drop('y', axis=1)
    y = df['y']

    # Convert target to binary (yes=1, no=0)
    y = (y == 'yes').astype(int)

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Exclude 'duration' feature (not realistic for prediction)
    if 'duration' in numerical_cols:
        print("WARNING: 'duration' feature is present. Excluding for realistic model.")
        numerical_cols.remove('duration')
        X = X.drop('duration', axis=1)

    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Numerical features: {len(numerical_cols)}")

    return X, y, categorical_cols, numerical_cols

def create_preprocessing_pipeline(X, categorical_cols, numerical_cols):
    """Create and fit preprocessing transformations"""
    X_processed = X.copy()
    label_encoders = {}

    # Encode categorical variables
    for col in categorical_cols:
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Scale numerical features
    scaler = StandardScaler()
    X_processed[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    preprocessing_artifacts = {"label_encoders": label_encoders, "scaler": scaler, "categorical_cols": categorical_cols,
                               "numerical_cols": numerical_cols, "feature_names": X_processed.columns.tolist()}

    return X_processed, preprocessing_artifacts

def preprocess_single_input(input_data, preprocessing):
    """Preprocess a single input sample for prediction"""
    df = pd.DataFrame([input_data])

    # Encode categorical variables
    for col, encoder in preprocessing["label_encoders"].items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col].astype(str))
            # Handle unseen categories
            except ValueError:
                df[col] = encoder.transform([encoder.classes_[0]])[0]

    # Scale numerical features
    df[preprocessing["numerical_cols"]] = preprocessing["scaler"].transform(df[preprocessing["numerical_cols"]])

    # Ensure correct feature order
    df = df[preprocessing["feature_names"]]
    return df


# DATA SPLITTING
def split_data(X, y, test_size, validation_size, random_state):
    """
    Split data into:
    - Training set (for model training)
    - Validation set (static, for evaluation)
    - Reference baseline (for monitoring baseline statistics)
    """
    print("\nSplitting data...")

    # First split: separate out test set
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Second split: separate validation from training
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=validation_size/(1-test_size),
                                                      random_state=random_state, stratify=y_temp)

    # Use test set as reference baseline
    X_baseline, y_baseline = X_test, y_test

    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Reference baseline: {len(X_baseline)} samples")

    return X_train, X_val, X_baseline, y_train, y_val, y_baseline


# MODEL TRAINING
def train_baseline_model(X_train, y_train):
    """Train logistic regression baseline model"""
    print("\nTraining logistic regression model...")

    model = LogisticRegression(max_iter=1000, random_state=CONFIG["random_state"], class_weight='balanced')
    model.fit(X_train, y_train)

    print("Model training completed.")
    return model


# MODEL EVALUATION
def evaluate_model(model, X, y, dataset_name=""):
    """Compute comprehensive evaluation metrics"""
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    metrics = {"accuracy": accuracy_score(y, y_pred), "precision": precision_score(y, y_pred), "recall": recall_score(y, y_pred),
               "f1_score": f1_score(y, y_pred), "roc_auc": roc_auc_score(y, y_pred_proba), "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
               "dataset_name": dataset_name, "n_samples": len(y), "positive_rate": y.mean()}

    print(f"\n{dataset_name} Metrics:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")

    return metrics

def compute_feature_statistics(X, y, model):
    """Compute baseline feature statistics for monitoring"""
    feature_stats = {}

    for col in X.columns:
        feature_stats[col] = {"mean": float(X[col].mean()), "std": float(X[col].std()), "min": float(X[col].min()),
                              "max": float(X[col].max()), "median": float(X[col].median()),
                              "q25": float(X[col].quantile(0.25)), "q75": float(X[col].quantile(0.75))}

    # Prediction distribution statistics
    y_pred_proba = model.predict_proba(X)[:, 1]

    prediction_stats = {"mean": float(y_pred_proba.mean()), "std": float(y_pred_proba.std()), "min": float(y_pred_proba.min()),
                        "max": float(y_pred_proba.max()), "median": float(np.median(y_pred_proba)),
                        "q25": float(np.quantile(y_pred_proba, 0.25)), "q75": float(np.quantile(y_pred_proba, 0.75))}

    return {"feature_statistics": feature_stats, "prediction_statistics": prediction_stats}


# SAVE AND LOAD ARTIFACTS
def save_artifacts(model, preprocessing_artifacts, baseline_metrics, baseline_stats, validation_metrics):
    """Save model, preprocessing pipeline, and baseline statistics"""

    # Create directories
    Path(CONFIG["model_dir"]).mkdir(exist_ok=True)
    Path(CONFIG["baseline_dir"]).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model
    model_path = f"{CONFIG['model_dir']}/model_v1.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

    # Save preprocessing artifacts
    preprocessing_path = f"{CONFIG['model_dir']}/preprocessing.pkl"
    joblib.dump(preprocessing_artifacts, preprocessing_path)
    print(f"Preprocessing artifacts saved to: {preprocessing_path}")

    # Save baseline reference
    baseline_data = {"version": "v1", "created_at": timestamp, "baseline_metrics": baseline_metrics,
                     "baseline_statistics": baseline_stats, "validation_metrics": validation_metrics,
                     "model_path": model_path, "preprocessing_path": preprocessing_path}

    baseline_path = f"{CONFIG['baseline_dir']}/reference_baseline.json"
    with open(baseline_path, 'w') as f:
        json.dump(baseline_data, f, indent=2)
    print(f"Baseline reference saved to: {baseline_path}")

    return model_path, preprocessing_path, baseline_path

def load_model_artifacts():
    """Load model and preprocessing pipeline"""
    model_path = f"{CONFIG['model_dir']}/model_v1.pkl"
    preprocessing_path = f"{CONFIG['model_dir']}/preprocessing.pkl"

    model = joblib.load(model_path)
    preprocessing = joblib.load(preprocessing_path)

    print("Model and preprocessing artifacts loaded successfully!")
    return model, preprocessing


# PREDICTION FUNCTION
def make_prediction(model, preprocessing, input_data):
    """
    Make a prediction on a single input.
    Args:
        model: Trained model
        preprocessing: Preprocessing artifacts
        input_data: Dict with feature values    
    Returns: Dict with prediction and probabilities
    """
    # Preprocess input
    X_processed = preprocess_single_input(input_data, preprocessing)

    # Predict
    prediction = int(model.predict(X_processed)[0])
    prediction_proba = model.predict_proba(X_processed)[0].tolist()

    result = {"prediction": prediction, "prediction_label": "yes" if prediction == 1 else "no",
              "prediction_proba": {"no": prediction_proba[0], "yes": prediction_proba[1]},
              "confidence": max(prediction_proba), "processed_features": X_processed.iloc[0].to_dict()}

    return result


# TRAINING PIPELINE
def train_pipeline():
    """Main training pipeline - runs everything"""

    print("="*80)
    print("STAGE 0: BASELINE MODEL TRAINING")
    print("="*80)

    # Step 1: Load and preprocess data
    X, y, categorical_cols, numerical_cols = load_and_preprocess_data(CONFIG["data_path"])
    X_processed, preprocessing_artifacts = create_preprocessing_pipeline(X, categorical_cols, numerical_cols)

    # Step 2: Split data
    X_train, X_val, X_baseline, y_train, y_val, y_baseline = split_data(X_processed, y, CONFIG["test_size"], CONFIG["validation_size"], CONFIG["random_state"])

    # Step 3: Train model
    model = train_baseline_model(X_train, y_train)

    # Step 4: Evaluate model
    validation_metrics = evaluate_model(model, X_val, y_val, "Validation Set")
    baseline_metrics = evaluate_model(model, X_baseline, y_baseline, "Reference Baseline")

    # Step 5: Compute baseline statistics
    baseline_stats = compute_feature_statistics(X_baseline, y_baseline, model)

    # Step 6: Save everything
    save_artifacts(model, preprocessing_artifacts, baseline_metrics, baseline_stats, validation_metrics)

    print("\n" + "="*80)
    print("STAGE 0 COMPLETE!")
    print("="*80)
    print("\nWhat we have:")
    print("  ✓ Trained logistic regression model (Model v1)")
    print("  ✓ Static validation dataset for evaluation")
    print("  ✓ Reference baseline with feature & prediction statistics")
    print("  ✓ Preprocessing pipeline saved")
    print("\nNext steps:")
    print("  1. Start the API: python main.py")
    print("  2. API will be available at: http://localhost:5000")


# MAIN
if __name__ == "__main__":
    train_pipeline()