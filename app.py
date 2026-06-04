# ===============================
# STREAMLIT APP - MUGILIDAE FISH CLASSIFIER
# LOAD MODELS & PREDICTION ONLY
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Mugilidae Fish Classifier", page_icon="🐟", layout="wide")

st.title("🐟 Mugilidae Fish Classification System")
st.markdown("### Identify 5 Mullet Species from Measurements")
st.markdown("---")

# Load models
@st.cache_resource
def load_models():
    models = {
        'ann': joblib.load('ann_model.pkl'),
        'pso': joblib.load('pso_model.pkl'),
        'ga': joblib.load('ga_model.pkl'),
        'gwo': joblib.load('gwo_model.pkl'),
        'scaler': joblib.load('scaler.pkl'),
        'label_encoder': joblib.load('label_encoder.pkl'),
        'feature_names': joblib.load('feature_names.pkl')
    }
    return models

models = load_models()

FEATURE_NAMES = models['feature_names']
FEATURE_DISPLAY = [
    "ND1_Total", "ND2_Total", "NP", "NC", "NV_Total", "NA_Total",
    "SL (mm)", "PL (mm)", "BH (mm)", "HL (mm)",
    "Head_Truss (mm)", "Anterior_Truss (mm)", "Mid_Truss (mm)",
    "Posterior_Truss (mm)", "Tail_Truss (mm)"
]

# Sidebar
st.sidebar.title("📋 About")
st.sidebar.info("""
**5 Mugilidae Species:**
- Planiliza subviridis
- Moolgarda seheli
- Osteomugil perusii
- Moolgarda tade
- Ellochelon vaigiensis

**15 Features:**
- Meristic (6)
- Morphometric (4)
- Truss (5)

**Models Available:**
- ANN (Baseline)
- ANN-PSO
- ANN-GA
- ANN-GWO
""")

# Model selection
model_choice = st.sidebar.selectbox(
    "Select Model for Prediction",
    ["ANN", "PSO", "GA", "GWO"]
)

st.sidebar.caption("FYP Project | UMT")

# Input form
st.header("🔮 Enter Fish Measurements")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Meristic Features")
    nd1 = st.number_input("ND1_Total", value=4.0, step=1.0)
    nd2 = st.number_input("ND2_Total", value=7.0, step=1.0)
    np_val = st.number_input("NP", value=14.0, step=1.0)
    nc = st.number_input("NC", value=14.0, step=1.0)
    nv = st.number_input("NV_Total", value=6.0, step=1.0)
    na = st.number_input("NA_Total", value=10.0, step=1.0)

with col2:
    st.subheader("Morphometric Features (mm)")
    sl = st.number_input("SL", value=150.0, step=10.0)
    pl = st.number_input("PL", value=40.0, step=5.0)
    bh = st.number_input("BH", value=45.0, step=5.0)
    hl = st.number_input("HL", value=40.0, step=5.0)

with col3:
    st.subheader("Truss Features (mm)")
    head = st.number_input("Head_Truss", value=80.0, step=10.0)
    ant = st.number_input("Anterior_Truss", value=70.0, step=10.0)
    mid = st.number_input("Mid_Truss", value=200.0, step=20.0)
    post = st.number_input("Posterior_Truss", value=200.0, step=20.0)
    tail = st.number_input("Tail_Truss", value=100.0, step=10.0)

if st.button("🔍 Predict Species", type="primary"):
    features = np.array([[nd1, nd2, np_val, nc, nv, na, sl, pl, bh, hl, head, ant, mid, post, tail]])
    features_scaled = models['scaler'].transform(features)
    
    if model_choice == "ANN":
        model = models['ann']
    elif model_choice == "PSO":
        model = models['pso']
    elif model_choice == "GA":
        model = models['ga']
    else:
        model = models['gwo']
    
    pred = model.predict(features_scaled)[0]
    species = models['label_encoder'].inverse_transform([pred])[0]
    proba = model.predict_proba(features_scaled)[0]
    confidence = max(proba) * 100
    
    st.markdown("---")
    st.success(f"### 🎯 Predicted Species: **{species}**")
    st.progress(int(confidence))
    st.caption(f"Confidence: {confidence:.1f}%")
    
    # Show all probabilities
    st.subheader("📊 Species Probabilities")
    prob_df = pd.DataFrame({
        'Species': models['label_encoder'].classes_,
        'Probability': proba
    }).sort_values('Probability', ascending=False)
    st.bar_chart(prob_df.set_index('Species'))

st.markdown("---")
st.caption("FYP Project | Universiti Malaysia Terengganu")
