import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from PIL import Image

# --- Page Config ---
st.set_page_config(
    page_title="Crime Early Warning System",
    page_icon="🚨",
    layout="wide",
)

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        .main {
            background-color: #f9fafc;
        }
        h1, h2, h3 {
            color: #003366;
        }
        .stButton>button {
            background-color: #004080;
            color: white;
            border-radius: 10px;
            padding: 0.6em 1.2em;
            border: none;
        }
        .stButton>button:hover {
            background-color: #0059b3;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load Models & Artifacts ---
MODEL_PATH = Path('rf_high_risk_model.joblib')
SCALER_PATH = Path('scaler.joblib')
LE_STATE_PATH = Path('le_state.joblib')
LE_DIST_PATH = Path('le_district.joblib')
FEATURES_PATH = Path('model_features.joblib')
TARGET_PATH = Path('model_target.joblib')
DATA_PATH = 'districtwise-crime-against-sts-2017-onwards.csv'

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    le_state = joblib.load(LE_STATE_PATH)
    le_district = joblib.load(LE_DIST_PATH)
    features = joblib.load(FEATURES_PATH)
    target = joblib.load(TARGET_PATH)
    return model, scaler, le_state, le_district, features, target

model, scaler, le_state, le_district, FEATURES, TARGET = load_artifacts()
df_base = pd.read_csv(DATA_PATH)

# --- App Header ---
st.title("🚨 Crime Early Warning System")
st.markdown("""
Predict and visualize **district-level crime risk** using historical trends.  
Use this as a decision-support dashboard for proactive policing and safety planning.
""")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📂 Batch Upload", "📊 Insights & Model Info"])

# --- Helper Function ---
def make_features_for_input(state, district, year, df=df_base):
    tmp = df[(df['state_name'] == state) & (df['district_name'] == district)].sort_values('year')
    lag1 = tmp[TARGET].iloc[-1] if tmp.shape[0] >= 1 else 0
    lag2 = tmp[TARGET].iloc[-2] if tmp.shape[0] >= 2 else 0
    state_enc = le_state.transform([state])[0] if state in le_state.classes_ else -1
    district_enc = le_district.transform([district])[0] if district in le_district.classes_ else -1

    feat = pd.DataFrame(np.zeros((1, len(FEATURES))), columns=FEATURES)
    if 'lag1' in FEATURES: feat['lag1'] = lag1
    if 'lag2' in FEATURES: feat['lag2'] = lag2
    if 'year' in FEATURES: feat['year'] = year
    if 'state_enc' in FEATURES: feat['state_enc'] = state_enc
    if 'district_enc' in FEATURES: feat['district_enc'] = district_enc

    # Fill crime columns with recent values if exist
    for c in df.columns:
        if c in FEATURES and c not in ['state_name', 'district_name', 'year', TARGET]:
            feat[c] = tmp[c].iloc[-1] if c in tmp.columns and tmp.shape[0] > 0 else 0

    return feat

# --- TAB 1: Single Prediction ---
with tab1:
    st.subheader("🔎 Predict High-Risk Crime Probability for a District")

    col1, col2, col3 = st.columns(3)
    with col1:
        state_sel = st.selectbox("Select State", sorted(df_base['state_name'].unique()))
    with col2:
        district_sel = st.selectbox(
            "Select District", 
            sorted(df_base[df_base['state_name'] == state_sel]['district_name'].unique())
        )
    with col3:
        year_sel = st.number_input("Select Year", 
                                   min_value=int(df_base['year'].min()), 
                                   max_value=int(df_base['year'].max()), 
                                   value=int(df_base['year'].max()))

    if st.button("🚀 Predict Now"):
        feat = make_features_for_input(state_sel, district_sel, year_sel)
        feat_scaled = scaler.transform(feat)
        prob = model.predict_proba(feat_scaled)[:, 1][0]
        label = model.predict(feat_scaled)[0]

        st.success(f"✅ Prediction complete for {district_sel}, {state_sel}")
        colA, colB = st.columns(2)
        colA.metric("Predicted Probability (High Risk)", f"{prob*100:.2f}%")
        colB.metric("Predicted Label", "🔴 High Risk" if label == 1 else "🟢 Low Risk")
        st.caption("Prediction based on historical trends and encoded district features.")
        st.dataframe(feat.T.style.background_gradient(cmap="Blues"), height=400)

# --- TAB 2: Batch Upload ---
with tab2:
    st.subheader("📂 Upload a CSV for Batch Predictions")
    uploaded = st.file_uploader("Upload file (CSV)", type=["csv"])
    if uploaded:
        df_upload = pd.read_csv(uploaded)
        rows = []
        for _, r in df_upload.iterrows():
            try:
                feat = make_features_for_input(r['state_name'], r['district_name'], int(r['year']))
                rows.append(feat.iloc[0])
            except:
                pass

        batch_X = pd.DataFrame(rows, columns=FEATURES)
        batch_scaled = scaler.transform(batch_X)
        probs = model.predict_proba(batch_scaled)[:, 1]
        preds = model.predict(batch_scaled)
        df_upload['pred_prob_high_risk'] = probs
        df_upload['pred_label_high_risk'] = preds

        st.success("✅ Batch predictions complete!")
        st.dataframe(df_upload.head(30))
        st.download_button("💾 Download Predictions", df_upload.to_csv(index=False), "predictions.csv", "text/csv")

# --- TAB 3: Insights ---
with tab3:
    st.subheader("📊 Model Insights and Feature Importance")
    st.write(f"Target variable: **{TARGET}**")
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    st.bar_chart(importances.head(15))
    st.caption("Top 15 features that contribute most to the risk prediction.")

    st.markdown("""
    **Usage Notes:**
    - Integrate socio-economic data for better model accuracy.
    - Retrain with recent years’ data regularly.
    - Combine this tool with field data for actionable intelligence.
    """)

    st.info("Built using Streamlit + Scikit-learn + Pandas + Joblib")

