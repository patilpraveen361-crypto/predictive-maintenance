import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/ai4i2020.csv")

# Remove columns that should not be used for prediction
df = df.drop(columns=[
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
])

# Separate features and target
X = df.drop(columns=["Machine failure"])
y = df["Machine failure"]

print("Features:")
print(X.columns.tolist())

print("\nTarget:")
print(y.name)

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)