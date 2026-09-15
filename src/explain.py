import pandas as pd
import joblib
import shap

# Load trained model
model = joblib.load("models/predictive_maintenance_model.pkl")

# Load dataset
df = pd.read_csv("data/raw/ai4i2020.csv")

# Remove unnecessary/leaky columns
df = df.drop(columns=[
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
])

# Convert machine type
df = pd.get_dummies(df, columns=["Type"], drop_first=True)

# Separate features and target
X = df.drop(columns=["Machine failure"])
y = df["Machine failure"]

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Explain first 100 test examples
X_sample = X.iloc[:100]

shap_values = explainer.shap_values(X_sample)

print("SHAP explanation created successfully!")

print("\nFeature names:")
print(X.columns.tolist())

print("\nSHAP values shape:")

if isinstance(shap_values, list):
    print([values.shape for values in shap_values])
else:
    print(shap_values.shape)