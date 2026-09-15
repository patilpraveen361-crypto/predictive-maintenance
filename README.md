\# ⚙️ Predictive Maintenance Data Science Pipeline



An end-to-end machine learning project that predicts industrial machine failures using sensor data and provides explainable AI insights with SHAP.



\## 🚀 Project Overview



Predictive maintenance uses machine sensor data to identify equipment that may fail before the failure occurs.



This project uses the \*\*AI4I 2020 Predictive Maintenance Dataset\*\* to build a Random Forest classification model that predicts whether a machine is likely to fail.



The project also includes an interactive \*\*Streamlit dashboard\*\* and \*\*SHAP-based explainability\*\* to show why the model made a particular prediction.



\## 🎯 Objectives



\- Predict machine failure from sensor measurements

\- Handle highly imbalanced failure data

\- Evaluate the model using classification metrics

\- Provide failure probability and risk classification

\- Explain individual predictions using SHAP

\- Build an interactive machine-monitoring dashboard



\## 📊 Dataset



The project uses the AI4I 2020 Predictive Maintenance Dataset.



The dataset contains:



\- Air temperature

\- Process temperature

\- Rotational speed

\- Torque

\- Tool wear

\- Machine type

\- Machine failure



The dataset contains \*\*10,000 machine records\*\*.



Machine failure is highly imbalanced, with approximately \*\*3.39% failure cases\*\*.



\## 🧠 Machine Learning Pipeline



```text

Raw Sensor Data

&#x20;      ↓

Data Cleaning

&#x20;      ↓

Feature Engineering

&#x20;      ↓

Train/Test Split

&#x20;      ↓

Random Forest Classifier

&#x20;      ↓

Failure Probability

&#x20;      ↓

Risk Classification

&#x20;      ↓

SHAP Explainability

&#x20;      ↓

Streamlit Dashboard

