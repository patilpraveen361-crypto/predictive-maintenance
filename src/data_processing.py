import pandas as pd

# Load raw dataset
df = pd.read_csv("data/raw/ai4i2020.csv")

# Display basic information
print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["Machine failure"].value_counts())