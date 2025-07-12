# app/streamlit_app.py

import sys
import os

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import uuid
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import cv2

# Import your *refactored* scripts
from scripts.preprocess_utils import preprocess_ppg_signal
from scripts.filter_utils import is_ppg_signal_acceptable
from scripts.segment_utils import segment_ppg_abp
from scripts.scalogram_utils import generate_scalogram_image
from scripts.scalogram_filter_utils import is_scalogram_acceptable
from scripts.model_utils import load_bp_model_and_scaler
from scripts.predict_utils import predict_bp

# ---------------- CONFIG ----------------
MODEL_PATH = "./models/best_hybrid_model.h5"
SCALER_PATH = "./models/label_scaler.save"
USER_DATA_ROOT = "./user_data_upload"

# Load model and scaler once
model, scaler = load_bp_model_and_scaler(MODEL_PATH, SCALER_PATH)

# Create session folder
session_id = str(uuid.uuid4())[:8]
session_folder = os.path.join(USER_DATA_ROOT, session_id)
os.makedirs(session_folder, exist_ok=True)

# Streamlit UI
st.set_page_config(page_title="BP Estimation from PPG", layout="centered")
st.title("💓 Blood Pressure Estimator")
st.markdown(
    """
    Upload your PPG+ABP CSV (columns: **PPG**, **ABP**).  
    The app will preprocess, segment, generate scalograms, and predict your **SBP/DBP**.  
    """
)

uploaded_file = st.file_uploader("📂 Upload your PPG CSV file", type=["csv"])

if uploaded_file:
    st.success("✅ File uploaded successfully!")
    # Save to user folder
    user_raw_csv = os.path.join(session_folder, "uploaded_ppg.csv")
    with open(user_raw_csv, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load CSV
    df = pd.read_csv(user_raw_csv)
    if 'PPG' not in df.columns or 'ABP' not in df.columns:
        st.error("❌ CSV must have columns: 'PPG' and 'ABP'")
        st.stop()

    # Extract columns
    ppg_raw = df['PPG'].values
    abp_raw = df['ABP'].values

    st.subheader("🔎 Preprocessing")
    st.write("Bandpass filtering, wavelet denoising, baseline removal, normalization...")

    ppg_preprocessed = preprocess_ppg_signal(ppg_raw)
    st.success(f"✅ Preprocessing complete. Length: {len(ppg_preprocessed)} samples.")

    # Quality filtering
    st.subheader("🩺 Quality Check")
    passed, reason = is_ppg_signal_acceptable(ppg_preprocessed)
    if not passed:
        st.error(f"❌ Signal rejected: {reason}")
        st.stop()
    else:
        st.success("✅ Signal passed quality checks!")

    # Segmentation
    st.subheader("✂️ Segmenting Signal")
    X_segments, Y_labels = segment_ppg_abp(ppg_preprocessed, abp_raw)
    if len(X_segments) == 0:
        st.error("❌ No valid segments found. Try a different recording.")
        st.stop()
    st.success(f"✅ Segmentation produced {len(X_segments)} valid segments.")

    # Select 1 segment (or average later)
    segment_idx = 0
    selected_ppg_segment = X_segments[segment_idx].reshape(-1, 1)

    # Scalogram generation
    st.subheader("🎨 Generating Scalogram")
    scalogram_img = generate_scalogram_image(X_segments[segment_idx])
    scalogram_rgb = cv2.cvtColor(scalogram_img, cv2.COLOR_GRAY2RGB)
    pil_img = Image.fromarray(scalogram_rgb)
    st.image(pil_img, caption="Generated Scalogram", width=300)

    # Scalogram quality filtering
    st.subheader("🧪 Scalogram Quality Check")
    acceptable, reason = is_scalogram_acceptable(scalogram_rgb)
    if not acceptable:
        st.error(f"❌ Scalogram rejected: {reason}")
        st.stop()
    else:
        st.success("✅ Scalogram passed quality filtering.")

    # ---- Model Prediction ----
    st.subheader("🤖 Predicting BP")
    try:
        predicted_sbp, predicted_dbp = predict_bp(
            model,
            scaler,
            scalogram_img,
            selected_ppg_segment
        )
        st.success(f"✅ Predicted SBP: {predicted_sbp:.2f} mmHg")
        st.success(f"✅ Predicted DBP: {predicted_dbp:.2f} mmHg")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

    # Interpret BP category   
    def bp_category(sbp, dbp):
        if sbp < 120 and dbp < 80:
            return "Normal"
        elif 120 <= sbp < 130 and dbp < 80:
            return "Elevated (Prehypertension)"
        elif 130 <= sbp < 140 or 80 <= dbp < 90:
            return "Hypertension Stage 1"
        elif sbp >= 140 or dbp >= 90:
            return "Hypertension Stage 2"
        else:
            return "Unknown"

    category = bp_category(predicted_sbp, predicted_dbp)
    st.subheader(f"🩸 BP Category: **{category}**")

    st.markdown("---")
    st.caption(f"Session ID: {session_id}")
    st.info("⚠️ Note: This app does not store your medical data permanently. All uploaded data is kept in a temporary folder per session.")

else:
    st.warning("📌 Please upload a CSV file with PPG and ABP columns to get started.")
