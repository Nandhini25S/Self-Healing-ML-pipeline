"""
Stage 1: Monitoring & Drift Detection

1. Read inference logs from Stage 0
2. Aggregate recent predictions into rolling window
3. Load baseline stats
4. Compare features against baseline
5. Compare predictions against baseline
6. Compute drift metrics
7. Decide healthy or degraded
8. Save report to disk
"""

import json
import numpy as np
from pathlib import Path
from stage0 import CONFIG
from datetime import datetime

MONITORING_CONFIG = {"rolling_window_size": 50, "feature_drift_threshold": 3.0, "prediction_drift_threshold": 0.1,
                     "monitoring_dir": "monitoring", "check_every_n_predictions": 100}

NUMERICAL_FEATURES = ["age", "campaign", "pdays", "previous", "emp.var.rate",
                      "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]

DATASET_CONFIG = {"dataset_dir": "datasets", "dataset_name": None}


def read_inference_logs():
    """Read all inference logs, skip any error entries"""
    log_path = Path(CONFIG["logs_dir"]) / "inference.log"
    if not log_path.exists():
        return []

    logs = []
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if "error" not in entry:
                    logs.append(entry)
            except json.JSONDecodeError:
                continue

    return logs

def get_rolling_window(logs):
    """Get the last N predictions from logs"""
    size = MONITORING_CONFIG["rolling_window_size"]
    return logs[-size:]

def load_baseline():
    """Load baseline statistics saved by Stage 0"""
    baseline_path = Path(CONFIG["baseline_dir"]) / "reference_baseline.json"
    with open(baseline_path, 'r') as f:
        data = json.load(f)
    return data["baseline_statistics"]

def compute_current_stats(window):
    """From the rolling window logs, extract numerical feature values and prediction probabilities. Compute their statistics"""
    feature_values = {feat: [] for feat in NUMERICAL_FEATURES}
    prediction_probas = []

    for log in window:
        input_feats = log["input_features"]

        # Collect numerical feature values
        for feat in NUMERICAL_FEATURES:
            if feat in input_feats:
                feature_values[feat].append(input_feats[feat])

        # Collect prediction probability
        prediction_probas.append(log["prediction_proba"])

    # Compute stats per feature
    feature_stats = {}
    for feat, values in feature_values.items():
        if not values:
            continue
        arr = np.array(values, dtype=float)
        feature_stats[feat] = {"mean": float(arr.mean()), "std": float(arr.std()),
            "min": float(arr.min()), "max": float(arr.max()), "median": float(np.median(arr))}

    # Compute prediction stats
    pred_arr = np.array(prediction_probas, dtype=float)
    prediction_stats = {"mean": float(pred_arr.mean()), "std": float(pred_arr.std()),
        "min": float(pred_arr.min()), "max": float(pred_arr.max()), "median": float(np.median(pred_arr))}

    return {"feature_statistics": feature_stats, "prediction_statistics": prediction_stats, "n_samples": len(window)}

def compute_drift(baseline, current_stats):
    """
    Compare current stats against baseline
    Feature drift:
        relative_change = |current_mean - baseline_mean| / baseline_std
        flagged if relative_change > 3
    Prediction drift:
        prediction_shift = |current_pred_mean - baseline_pred_mean|
        flagged if prediction_shift > 0.1
    """
    baseline_features = baseline["feature_statistics"]
    current_features = current_stats["feature_statistics"]

    # --- Feature drift ---
    feature_drift = {}
    drifted_features = []

    for feat in current_features:
        if feat not in baseline_features:
            continue

        b_mean = baseline_features[feat]["mean"]
        b_std = baseline_features[feat]["std"]
        c_mean = current_features[feat]["mean"]

        # Avoid division by zero
        if b_std == 0:
            b_std = 1e-6

        relative_change = abs(c_mean - b_mean) / b_std
        is_drifted = relative_change > MONITORING_CONFIG["feature_drift_threshold"]

        feature_drift[feat] = {"baseline_mean": b_mean, "baseline_std": b_std, "current_mean": c_mean,
                               "relative_change": round(relative_change, 4), "is_drifted": is_drifted}

        if is_drifted:
            drifted_features.append(feat)

    # --- Prediction drift ---
    b_pred_mean = baseline["prediction_statistics"]["mean"]
    c_pred_mean = current_stats["prediction_statistics"]["mean"]
    prediction_shift = abs(c_pred_mean - b_pred_mean)
    prediction_drifted = prediction_shift > MONITORING_CONFIG["prediction_drift_threshold"]

    prediction_drift = {"baseline_mean": round(b_pred_mean, 4), "current_mean": round(c_pred_mean, 4),
                        "prediction_shift": round(prediction_shift, 4), "is_drifted": prediction_drifted}

    return {"feature_drift": feature_drift, "drifted_features": drifted_features, "prediction_drift": prediction_drift}

def determine_status(drift_results):
    """
    If any feature drifted OR prediction drifted → DEGRADED
    Otherwise → HEALTHY
    """
    if drift_results["drifted_features"] or drift_results["prediction_drift"]["is_drifted"]:
        return "DEGRADED"
    return "HEALTHY"

def save_report(status, drift_results, current_stats):
    """Save the drift report as a JSON file"""
    Path(MONITORING_CONFIG["monitoring_dir"]).mkdir(exist_ok=True)
    timestamp = datetime.now()

    report = {
        "timestamp": timestamp.isoformat(), "status": status,
        "n_samples_analyzed": current_stats["n_samples"],
        "thresholds": {
            "feature_drift_threshold": MONITORING_CONFIG["feature_drift_threshold"],
            "prediction_drift_threshold": MONITORING_CONFIG["prediction_drift_threshold"]},
        "summary": {
            "n_drifted_features": len(drift_results["drifted_features"]),
            "drifted_features": drift_results["drifted_features"],
            "prediction_drifted": drift_results["prediction_drift"]["is_drifted"]},
        "feature_drift": drift_results["feature_drift"],
        "prediction_drift": drift_results["prediction_drift"]}

    # Save timestamped report
    filename = f"drift_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    report_path = Path(MONITORING_CONFIG["monitoring_dir"]) / filename
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Save as latest
    latest_path = Path(MONITORING_CONFIG["monitoring_dir"]) / "drift_report_latest.json"
    with open(latest_path, 'w') as f:
        json.dump(report, f, indent=2)

    return report


def run_monitoring_check():
    """
    Run the full monitoring pipeline:
    1 → Read logs
    2 → Rolling window
    3 → Load baseline
    4+5 → Compute current stats
    6 → Compute drift
    7 → Determine status
    8 → Save report
    """
    print("\n" + "="*60)
    print("STAGE 1: DRIFT DETECTION")
    print("="*60)

    # Step 1
    print("\n1. Reading inference logs...")
    logs = read_inference_logs()
    print(f"   Total predictions in log: {len(logs)}")

    if len(logs) == 0:
        print("   ⚠️  No predictions found. Make some predictions first.")
        return None

    # Step 2
    print(f"\n2. Rolling window (last {MONITORING_CONFIG['rolling_window_size']})...")
    window = get_rolling_window(logs)
    print(f"   Using {len(window)} predictions")

    # Step 3
    print("\n3. Loading baseline...")
    baseline = load_baseline()
    print("   ✓ Baseline loaded")

    # Step 4 & 5
    print("\n4. Computing current statistics...")
    current_stats = compute_current_stats(window)
    print(f"   ✓ Stats computed for {len(current_stats['feature_statistics'])} features")

    # Step 6
    print("\n5. Computing drift metrics...")
    drift_results = compute_drift(baseline, current_stats)
    print(f"   Features drifted: {drift_results['drifted_features'] or 'None'}")
    print(f"   Prediction drifted: {drift_results['prediction_drift']['is_drifted']}")

    # Step 7
    print("\n6. Determining status...")
    status = determine_status(drift_results)
    print(f"   Status: {status}")

    # Step 8
    print("\n7. Saving report...")
    report = save_report(status, drift_results, current_stats)
    print(f"   ✓ Saved to monitoring/")

    print("\n" + "="*60)
    print(f"  RESULT: {status}")
    print("="*60)

    return report

def get_latest_report():
    """Load and return the latest drift report. None if no report exists"""
    latest_path = Path(MONITORING_CONFIG["monitoring_dir"]) / "drift_report_latest.json"
    if not latest_path.exists():
        return None

    with open(latest_path, 'r') as f:
        return json.load(f)

def get_current_status():
    """Quick status check for the API"""
    logs = read_inference_logs()
    latest_report = get_latest_report()
    return {
        "status": latest_report["status"] if latest_report else "UNKNOWN", "total_predictions": len(logs),
        "last_check_timestamp": latest_report["timestamp"] if latest_report else None,
        "rolling_window_size": MONITORING_CONFIG["rolling_window_size"]}


if __name__ == "__main__":
    run_monitoring_check()