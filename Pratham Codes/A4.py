# assignment4_breast_cancer_rs.py
# Simple ML-based breast cancer prognosis recommender (for RS-A4 dataset)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Step 1: Load dataset
df = pd.read_csv("RS-A4_SEER Breast Cancer Dataset .csv")

# Step 2: Drop unused column
df = df.drop(columns=["Unnamed: 3"])

# Step 3: Define target and features
target = "Status"          # Alive / Dead
X = df.drop(columns=[target])
y = df[target]

# Step 4: Encode categorical columns
le_dict = {}
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        le_dict[col] = le

# Encode target column
le_target = LabelEncoder()
y = le_target.fit_transform(y)   # Alive→0, Dead→1 (or vice versa depending on order)

# Step 5: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(acc, 3))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 8: Recommendation function
def prognosis_recommendation(features):
    """
    Give prognosis recommendation based on model prediction.
    features: array of patient feature values (same order as X columns)
    """
    prediction = model.predict([features])
    # decode numeric to label string (Alive/Dead)
    label = le_target.inverse_transform(prediction)[0]
    if label.lower() == "dead":
        return "High risk detected. Immediate consultation and further tests recommended."
    else:
        return "Prognosis positive. Routine monitoring suggested, follow up with healthcare provider."

# Step 9: Example patient

example_patient = X_test.iloc[0].values
# Show the patient used for recommendation
print("\nExample patient details:")
print(X_test.iloc[0])
print("Actual Status:", le_target.inverse_transform([y_test[0]])[0])
recommendation = prognosis_recommendation(example_patient)
print("\nRecommendation:", recommendation)




# ===============================
# 🔍 Check prognosis for any patient
# ===============================

def check_patient(index):
    """
    Shows details, actual status, predicted status, and recommendation
    for the chosen patient index in X_test.
    Example: check_patient(10)
    """
    # show selected patient details
    print(f"\n=== Patient Index: {index} ===")
    patient = X_test.iloc[index]
    print("\nPatient Details:")
    print(patient)
    
    # actual status
    actual = le_target.inverse_transform([y_test[index]])[0]
    print("\nActual Status:", actual)
    
    # predicted status
    pred = model.predict([patient.values])
    predicted = le_target.inverse_transform(pred)[0]
    print("Predicted Status:", predicted)
    
    # recommendation message
    if predicted.lower() == "dead":
        msg = "⚠️ High risk detected. Immediate consultation and further tests recommended."
    else:
        msg = "✅ Prognosis positive. Routine monitoring suggested, follow up with healthcare provider."
    
    print("\nRecommendation:", msg)

# 💡 Example usage:
# check_patient(0)  # see first patient from test set
# check_patient(10) # see 10th patient from test set

