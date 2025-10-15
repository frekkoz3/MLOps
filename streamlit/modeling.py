
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# -------------------------------
# 1. Load Dataset
# -------------------------------
url = 'https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv'
df = pd.read_csv(url)

# Rename target column for consistency
df.rename(columns={'Class': 'is_fraud'}, inplace=True)

# -------------------------------
# 2. Feature and Target Split
# -------------------------------
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# -------------------------------
# 3. Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 4. Feature Scaling
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 5. Model Training
# -------------------------------
cls = RandomForestClassifier(
    n_estimators=200, random_state=42, n_jobs=-1
)
cls.fit(X_train_scaled, y_train)

# -------------------------------
# 6. Evaluation
# -------------------------------
y_pred = cls.predict(X_test_scaled)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# -------------------------------
# 7. Save Model and Scaler
# -------------------------------
with open('model.pkl', 'wb') as f:
    pickle.dump(cls, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n✅ Model and scaler saved (model.pkl, scaler.pkl)")
