import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Mugilidae Fish Classifier - Test", layout="wide")

st.title("🐟 Mugilidae Fish Classifier - TEST VERSION")

# Load models
@st.cache_resource
def load_models():
    try:
        model = joblib.load('gwo_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        return model, scaler, label_encoder
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

model, scaler, label_encoder = load_models()

if model is not None:
    st.success("✅ Model loaded successfully!")
    
    # Input fields - 15 features
    st.subheader("Enter 15 Measurements")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        nd1 = st.number_input("ND1_Total", value=4.0, step=1.0)
        nd2 = st.number_input("ND2_Total", value=6.0, step=1.0)
        np_val = st.number_input("NP", value=14.0, step=1.0)
        nc = st.number_input("NC", value=15.0, step=1.0)
        nv = st.number_input("NV_Total", value=6.0, step=1.0)
        na = st.number_input("NA_Total", value=11.0, step=1.0)
    
    with col2:
        sl = st.number_input("SL", value=529.23, step=10.0)
        pl = st.number_input("PL", value=210.91, step=5.0)
        bh = st.number_input("BH", value=22.57, step=5.0)
        hl = st.number_input("HL", value=216.36, step=5.0)
    
    with col3:
        head = st.number_input("Head_Truss", value=71.60, step=5.0)
        ant = st.number_input("Anterior_Truss", value=85.28, step=5.0)
        mid = st.number_input("Mid_Truss", value=280.68, step=10.0)
        post = st.number_input("Posterior_Truss", value=306.43, step=10.0)
        tail = st.number_input("Tail_Truss", value=228.54, step=10.0)
    
    if st.button("🔍 Predict", type="primary"):
        try:
            features = np.array([[nd1, nd2, np_val, nc, nv, na, sl, pl, bh, hl, 
                                  head, ant, mid, post, tail]])
            features_scaled = scaler.transform(features)
            pred = model.predict(features_scaled)[0]
            species = label_encoder.inverse_transform([pred])[0]
            proba = model.predict_proba(features_scaled)[0]
            confidence = max(proba) * 100
            
            st.success(f"### 🎯 Predicted Species: **{species}**")
            st.progress(int(confidence))
            st.caption(f"Confidence: {confidence:.1f}%")
            
            st.subheader("📊 Probabilities")
            prob_df = pd.DataFrame({
                'Species': label_encoder.classes_,
                'Probability': proba
            }).sort_values('Probability', ascending=False)
            st.bar_chart(prob_df.set_index('Species'))
            
        except Exception as e:
            st.error(f"Error: {e}")
