import time
import requests
import pandas as pd
import streamlit as st
from pathlib import Path

API_BASE = "http://127.0.0.1:5000"

st.set_page_config(page_title="Self-Healing ML Pipeline", page_icon="🤖", layout="wide")

# Session state
if "dataset_uploaded" not in st.session_state:
    st.session_state.dataset_uploaded = False
if "show_prediction_form" not in st.session_state:
    st.session_state.show_prediction_form = False
if "feature_schema" not in st.session_state:
    st.session_state.feature_schema = None

def upload_page():
    st.title("🤖 Self-Healing ML Pipeline")
    st.markdown("---")

    st.info("⚠️ **Important:** Last column → y (target), All other columns → X (features)")

    col1, col2 = st.columns([1, 1])

    with col1:
        dataset_name = st.text_input("Dataset Name", placeholder="e.g., customer_churn")

    with col2:
        uploaded_file = st.file_uploader("Upload Dataset", type=["xlsx", "csv"])

    if st.button("📤 Upload & Train Model", type="primary", disabled=not (dataset_name and uploaded_file)):
        with st.spinner("Uploading and training..."):
            # Save file temporarily
            temp_path = Path("temp_upload") / uploaded_file.name
            temp_path.parent.mkdir(exist_ok=True)

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Send to backend
            try:
                response = requests.post(
                    f"{API_BASE}/dataset/upload",
                    json={"dataset_name": dataset_name, "file_path": str(temp_path)}
                )

                if response.status_code == 200:
                    st.success("✅ Dataset uploaded successfully!")
                    st.session_state.dataset_uploaded = True
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Upload failed: {response.text}")

            except Exception as e:
                st.error(f"Error: {e}")

def dashboard_page():
    st.title("📊 Dashboard")
    st.markdown("---")

    # Get dataset info
    try:
        dataset_resp = requests.get(f"{API_BASE}/dataset/info")
        if dataset_resp.status_code == 200:
            dataset_info = dataset_resp.json()

            col1, col2, col3 = st.columns(3)
            col1.metric("Dataset", dataset_info["name"])
            col2.metric("Samples", dataset_info["n_samples"])
            col3.metric("Features", dataset_info["n_features"])
    except:
        pass

    # Model status
    try:
        status_resp = requests.get(f"{API_BASE}/monitoring/status")
        if status_resp.status_code == 200:
            status = status_resp.json()

            st.subheader("📊 Model Status")
            col1, col2 = st.columns(2)

            status_emoji = "🟢" if status["status"] == "HEALTHY" else "🔴"
            col1.metric("Status", f"{status_emoji} {status['status']}")
            col2.metric("Predictions Made", status["total_predictions"])
    except:
        st.warning("Model not trained yet")

    st.markdown("---")

    # Prediction section
    st.subheader("🔮 Make Prediction")

    if st.button("Open Prediction Form"):
        st.session_state.show_prediction_form = True
        # Fetch schema
        try:
            schema_resp = requests.get(f"{API_BASE}/features/schema")
            if schema_resp.status_code == 200:
                st.session_state.feature_schema = schema_resp.json()
        except Exception as e:
            st.error(f"Error fetching schema: {e}")

    if st.session_state.show_prediction_form and st.session_state.feature_schema:
        with st.form("prediction_form"):
            st.write("**Enter feature values:**")

            input_data = {}
            features = st.session_state.feature_schema["features"]

            cols = st.columns(2)
            for idx, feature in enumerate(features):
                col = cols[idx % 2]

                if feature["type"] == "number":
                    input_data[feature["name"]] = col.number_input(
                        feature["name"],
                        value=float(feature["sample_value"]) if feature["sample_value"] else 0.0
                    )
                else:
                    input_data[feature["name"]] = col.text_input(
                        feature["name"],
                        value=feature["sample_value"]
                    )

            submitted = st.form_submit_button("🚀 Predict", type="primary")

            if submitted:
                try:
                    pred_resp = requests.post(f"{API_BASE}/predict", json=input_data)
                    if pred_resp.status_code == 200:
                        result = pred_resp.json()
                        st.success(f"**Prediction:** {result['prediction_label']}")
                        st.write(f"**Confidence:** {result['confidence']:.2%}")
                        st.json(result["prediction_proba"])
                    else:
                        st.error(f"Prediction failed: {pred_resp.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")

    # Monitoring section
    st.subheader("📈 Monitoring")

    col1, col2, col3 = st.columns(3)

    if col1.button("🔍 Get Status"):
        try:
            resp = requests.get(f"{API_BASE}/monitoring/status")
            if resp.status_code == 200:
                st.json(resp.json())
        except Exception as e:
            st.error(f"Error: {e}")

    if col2.button("📄 Latest Report"):
        try:
            resp = requests.get(f"{API_BASE}/monitoring/latest-report")
            if resp.status_code == 200:
                st.json(resp.json())
            else:
                st.warning("No report available yet")
        except Exception as e:
            st.error(f"Error: {e}")

    if col3.button("⚡ Trigger Check"):
        with st.spinner("Running drift check..."):
            try:
                resp = requests.post(f"{API_BASE}/monitoring/trigger")
                if resp.status_code == 200:
                    st.success("Drift check completed!")
                    st.json(resp.json())
            except Exception as e:
                st.error(f"Error: {e}")

# Main routing
if not st.session_state.dataset_uploaded:
    upload_page()
else:
    dashboard_page()