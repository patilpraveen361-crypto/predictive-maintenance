import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load model
model = joblib.load("models/predictive_maintenance_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

# Title
st.title("⚙️ Predictive Maintenance System")
st.write("Industrial Machine Failure Prediction")

st.divider()

# Sensor inputs
st.subheader("Machine Sensor Data")

col1, col2 = st.columns(2)

with col1:
    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"]
    )

    air_temp = st.number_input(
        "Air Temperature [K]",
        min_value=250.0,
        max_value=350.0,
        value=300.0
    )

    process_temp = st.number_input(
        "Process Temperature [K]",
        min_value=250.0,
        max_value=400.0,
        value=310.0
    )

with col2:
    speed = st.number_input(
        "Rotational Speed [rpm]",
        min_value=500,
        max_value=3000,
        value=1500
    )

    torque = st.number_input(
        "Torque [Nm]",
        min_value=0.0,
        max_value=100.0,
        value=40.0
    )

    tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=0,
        max_value=300,
        value=100
    )

# Prediction button
if st.button("🔍 Predict Machine Failure"):

    machine = pd.DataFrame([{
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Type_L": machine_type == "L",
        "Type_M": machine_type == "M"
    }])

    # Match training features
    machine = machine.reindex(
        columns=feature_names,
        fill_value=False
    )

    # Prediction
    prediction = model.predict(machine)[0]
    probability = model.predict_proba(machine)[0][1]

    st.divider()

    st.subheader("Prediction Result")

    st.metric(
        "Failure Probability",
        f"{probability:.2%}"
    )

    # Risk classification
    if probability >= 0.70:
        st.error(
            "🔴 CRITICAL RISK — Immediate maintenance recommended."
        )

    elif probability >= 0.40:
        st.warning(
            "🟠 MEDIUM RISK — Schedule maintenance soon."
        )

    elif probability >= 0.20:
        st.info(
            "🟡 LOW RISK — Monitor machine condition."
        )

    else:
        st.success(
            "🟢 NORMAL — Machine condition appears stable."
        )    # SHAP explanation
    st.subheader("🔎 Why did the model make this prediction?")

    explainer = shap.TreeExplainer(model)
    shap_explanation = explainer(machine)

    shap_values = shap_explanation.values

    # For binary classification, select class 1 (failure)
    if len(shap_values.shape) == 3:
        shap_values = shap_values[0, :, 1]
    else:
        shap_values = shap_values[0]

    explanation_df = pd.DataFrame({
        "Feature": machine.columns,
        "SHAP Impact": shap_values,
        "Value": machine.iloc[0].values
    })

    explanation_df["Absolute Impact"] = explanation_df["SHAP Impact"].abs()

    explanation_df = explanation_df.sort_values(
        "Absolute Impact",
        ascending=False
    )

    st.dataframe(
        explanation_df[
            ["Feature", "Value", "SHAP Impact"]
        ],
        use_container_width=True
    )
    # SHAP impact chart
    chart_df = explanation_df.sort_values("SHAP Impact")

    fig, ax = plt.subplots()

    ax.barh(
        chart_df["Feature"],
        chart_df["SHAP Impact"]
    )

    ax.set_xlabel("SHAP Impact")
    ax.set_title("Feature Contribution to Failure Prediction")

    st.pyplot(fig)
    plt.close(fig)