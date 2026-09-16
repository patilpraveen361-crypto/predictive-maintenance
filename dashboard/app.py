import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM CSS STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance Control Center",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphic Dark Theme Styling
st.markdown("""
    <style>
    /* Dark Theme Background */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Modern Dashboard Cards */
    .css-card {
        background-color: #1E222D;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        border: 1px solid #2B313E;
        margin-bottom: 20px;
    }
    
    /* Status Badges */
    .status-normal {
        background-color: #1b4332;
        color: #4EBA6F;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    .status-warning {
        background-color: #4a2810;
        color: #FF9F43;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    .status-danger {
        background-color: #4a1212;
        color: #FF5252;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }

    /* Target standard Streamlit UI adjustments */
    div[data-testid="stSidebar"] {
        background-color: #161922;
        border-right: 1px solid #2B313E;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. LOAD TRAINED ASSETS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("models/predictive_maintenance_model.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, feature_names

model, feature_names = load_assets()

# -----------------------------------------------------------------------------
# 3. HEADER & HERO SECTION
# -----------------------------------------------------------------------------
st.title("⚙️ Industrial Predictive Maintenance Platform")
st.caption("Real-Time Machine Failure Risk Analysis & Explainable AI (SHAP) Diagnostics")
st.divider()

# -----------------------------------------------------------------------------
# 4. SIDEBAR - REAL-TIME SENSOR INPUTS
# -----------------------------------------------------------------------------
st.sidebar.header("🎛️ Live Sensor Parameters")
st.sidebar.markdown("Adjust the values below to simulate live telemetry from industrial equipment.")

input_data = {}
# Generate dynamic input UI based on model feature names
for feature in feature_names:
    input_data[feature] = st.sidebar.slider(
        label=f"{feature.replace('_', ' ').title()}",
        min_value=0.0,
        max_value=500.0,
        value=50.0,
        step=1.0
    )

input_df = pd.DataFrame([input_data])

# -----------------------------------------------------------------------------
# 5. INFERENCE & DASHBOARD DISPLAY
# -----------------------------------------------------------------------------
# Compute Model Prediction
prediction = model.predict(input_df)[0]
prediction_prob = model.predict_proba(input_df)[0][1]

# Top Metrics Overview Layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown("### Equipment Status")
    if prediction_prob < 0.3:
        st.markdown('<span class="status-normal">🟢 HEALTHY</span>', unsafe_allow_html=True)
    elif 0.3 <= prediction_prob < 0.7:
        st.markdown('<span class="status-warning">⚠️ MONITOR CLOSELY</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-danger">🚨 FAILURE IMMINENT</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown("### Calculated Failure Risk")
    st.metric(label="Probability", value=f"{prediction_prob * 100:.1f}%")
    st.progress(float(prediction_prob))
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown("### Model Decision")
    status_str = "Failure Detected" if prediction == 1 else "Normal Operation"
    st.metric(label="Predicted Class", value=status_str)
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# -----------------------------------------------------------------------------
# 6. EXPLAINABLE AI (SHAP DIAGNOSTICS)
# -----------------------------------------------------------------------------
st.subheader("📊 Explainable AI Diagnostics (SHAP Analysis)")
st.write("Understand which specific sensor metrics drove the machine failure risk calculation.")

tab1, tab2 = st.tabs(["📉 Waterfall Plot", "📋 Input Feature Table"])

with tab1:
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(input_df)

        # Slice for the failure class (class 1) and the single input sample
        if len(shap_values.shape) == 3:
            single_explanation = shap_values[0, :, 1]
        else:
            single_explanation = shap_values[0]

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        
        shap.plots.waterfall(single_explanation, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating SHAP plot: {e}")

with tab2:
    st.markdown("##### Current Sensor Readouts")
    st.dataframe(input_df, use_container_width=True)
