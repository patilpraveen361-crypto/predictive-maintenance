import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
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

# Convert machine type to numerical columns
df = pd.get_dummies(df, columns=["Type"], drop_first=True)

# Separate features and target
X = df.drop(columns=["Machine failure"])
y = df["Machine failure"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Random Forest model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Train
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

# Evaluation
print("\n===== RANDOM FOREST RESULTS =====")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nROC-AUC Score:")
print(round(roc_auc_score(y_test, y_probability), 4))
print("\nPR-AUC Score:")
print(round(average_precision_score(y_test, y_probability), 4))

# Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

# Save model
joblib.dump(
    model,
    "models/predictive_maintenance_model.pkl"
)

# Save feature names
joblib.dump(
    X.columns.tolist(),
    "models/feature_names.pkl"
)

print("\nRandom Forest model saved successfully!")