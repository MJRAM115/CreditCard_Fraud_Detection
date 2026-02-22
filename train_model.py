import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Load dataset
df = pd.read_csv("credit_card_fraud.csv")
df.columns = df.columns.str.strip()
df.drop_duplicates(inplace=True)

# Auto detect last column as target
target_col = df.columns[-1]
print("Target column detected:", target_col)

X = df.drop(columns=[target_col])
y = df[target_col]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(list(X.columns), open("columns.pkl", "wb"))

print("✅ Model saved successfully!")
