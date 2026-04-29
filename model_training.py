import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("Toddler Autism dataset July 2018 (1).csv")

# Clean Columns
df.drop(columns=['Case_No', 'Who completed the test'], inplace=True)
df.rename(columns={"Class/ASD Traits ": "ASD"}, inplace=True)

# Fill Missing Values
df.fillna(df.mode().iloc[0], inplace=True)

# -----------------------------
# 2. Encode Categorical Features
# -----------------------------
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# -----------------------------
# 3. Split Data
# -----------------------------
X = df.drop("ASD", axis=1)
y = df["ASD"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Scale Features
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# 5. Train the Best Model (Random Forest)
# -----------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# 6. Save Model, Scaler, Encoders
# -----------------------------
with open("best_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("Files saved: best_model.pkl, scaler.pkl, label_encoders.pkl")
