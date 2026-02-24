import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ----------------------------
# 1. Load Dataset
# ----------------------------
df = pd.read_csv("credit_card_fraud.csv")
df.columns = df.columns.str.strip()
df.drop_duplicates(inplace=True)

print("Dataset Shape:", df.shape)

# ----------------------------
# 2. Target & Features
# ----------------------------
target_col = df.columns[-1]
print("Target column detected:", target_col)

X = df.drop(columns=[target_col])
y = df[target_col]

print("Class Distribution:\n", y.value_counts())

# ----------------------------
# 3. Scaling
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# 4. Train-Test Split (IMPORTANT)
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y  # VERY IMPORTANT for fraud dataset
)

# ----------------------------
# 5. Model (Balanced Random Forest)
# ----------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",   # KEY FIX
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ----------------------------
# 6. Evaluation
# ----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

print("\nROC-AUC Score:", roc_auc_score(y_test, y_prob))

# ----------------------------
# 7. Save Model Files
# ----------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(list(X.columns), open("columns.pkl", "wb"))

print("\n✅ Model, Scaler, and Columns saved successfully!")
