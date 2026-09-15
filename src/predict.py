import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/predictive_maintenance_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# Example machine sensor data
machine = pd.DataFrame([{
    "Air temperature [K]": 300,
    "Process temperature [K]": 310,
    "Rotational speed [rpm]": 1500,
    "Torque [Nm]": 40,
    "Tool wear [min]": 100,
    "Type_L": False,
    "Type_M": True
}])

# Make sure columns match training data
machine = machine.reindex(columns=feature_names, fill_value=False)

# Predict
prediction = model.predict(machine)[0]
probability = model.predict_proba(machine)[0][1]

print("Machine Failure Prediction:", prediction)
print(f"Failure Probability: {probability:.2%}")

if prediction == 1:
    print("Status: HIGH RISK - Maintenance recommended")
else:
    print("Status: NORMAL - No immediate failure predicted")