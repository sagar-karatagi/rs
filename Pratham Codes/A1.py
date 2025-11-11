# assignment1_yield_and_reco.py
# Simple Yield Prediction + Recommendation System (no CNN)
# Author: Harsh Balkrishna Vahal

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# -----------------------
# 1. Load dataset
# -----------------------
DATA_PATH = "RS-A1_yield.csv"   # change path if needed
df = pd.read_csv(DATA_PATH)
print("Loaded:", DATA_PATH)
print("Shape:", df.shape)
print(df.head())

# -----------------------
# 2. Prepare features & target
# -----------------------
# We will use 'hg/ha_yield' as the target (as in your earlier file).
TARGET = "hg/ha_yield"

# If this exact column doesn't exist, stop and inform (so you can change it)
if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found in dataset. Change TARGET variable in script.")

# Choose sensible input features (explicit)
# We'll use rainfall, pesticides, avg_temp, Year, and categorical Area/Item if present.
feature_cols = []
for c in ["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp", "Year"]:
    if c in df.columns:
        feature_cols.append(c)

# include Area and Item as categorical if present
cat_cols = [c for c in ("Area", "Item") if c in df.columns]
print("\nUsing numeric features:", feature_cols)
print("Using categorical features:", cat_cols)

# Build X and y
X_num = df[feature_cols].copy() if feature_cols else pd.DataFrame(index=df.index)
X_cat = df[cat_cols].copy() if cat_cols else pd.DataFrame(index=df.index)
y = df[TARGET].astype(float)

# Simple encoding for categorical features (OneHot)
if not X_cat.empty:
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat_enc = pd.DataFrame(ohe.fit_transform(X_cat), index=X_cat.index,
                             columns=[f"{col}__{val}" for col, vals in zip(X_cat.columns, ohe.categories_) for val in vals])
else:
    X_cat_enc = pd.DataFrame(index=df.index)

# Combine
X = pd.concat([X_num.reset_index(drop=True), X_cat_enc.reset_index(drop=True)], axis=1)

# Fill any remaining NaNs with median (numeric) or 0
for col in X.columns:
    if X[col].dtype.kind in "biufc":
        X[col] = X[col].fillna(X[col].median())
    else:
        X[col] = X[col].fillna(0)

print("\nFinal feature matrix shape:", X.shape)

# -----------------------
# 3. Train/Test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)

# -----------------------
# 4. Train RandomForestRegressor
# -----------------------
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("\nModel trained.")

# -----------------------
# 5. Evaluate
# -----------------------

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # ✅ fixed
r2 = r2_score(y_test, y_pred)

print("\nEvaluation on test set:")
print("RMSE:", round(rmse, 4))
print("R2:", round(r2, 4))


# Save model and encoder
joblib.dump(model, "yield_model_rf.pkl")
if not X_cat.empty:
    joblib.dump(ohe, "yield_ohe.pkl")
print("\nSaved: yield_model_rf.pkl", ("yield_ohe.pkl" if not X_cat.empty else ""))

# -----------------------
# 6. Helper functions
# -----------------------
def predict_yield(input_dict):
    """
    input_dict: dictionary of input features. Example:
    {
      "average_rain_fall_mm_per_year": 300,
      "pesticides_tonnes": 2.5,
      "avg_temp": 27,
      "Year": 2024,
      "Area": "SomeArea",
      "Item": "Wheat"
    }
    Only provide the columns you intend to use; missing numeric keys will be filled with training medians.
    """
    # Build a single-row DataFrame matching X columns
    row_num = {}
    for c in feature_cols:
        row_num[c] = input_dict.get(c, X[c].median() if c in X.columns else 0)
    # categorical
    row_cat = {}
    for c in cat_cols:
        row_cat[c] = input_dict.get(c, "")
    # encode categorical if needed
    if not X_cat.empty:
        # transform with fitted ohe; unseen categories handled by ignore
        cat_df = pd.DataFrame([row_cat])
        cat_enc = pd.DataFrame(ohe.transform(cat_df), columns=[f"{col}__{val}" for col, vals in zip(X_cat.columns, ohe.categories_) for val in vals])
    else:
        cat_enc = pd.DataFrame()
    row = pd.concat([pd.DataFrame([row_num]).reset_index(drop=True), cat_enc.reset_index(drop=True)], axis=1)
    # ensure all columns present
    for col in X.columns:
        if col not in row.columns:
            row[col] = 0
    row = row[X.columns]  # reorder
    pred = model.predict(row)[0]
    return float(pred)

def advise(disease_detected, predicted_yield, yield_threshold=None):
    """
    Returns a list of textual recommendations.
    - disease_detected: bool (True if disease detected)
    - predicted_yield: float (from predict_yield)
    - yield_threshold: float or None. If None, threshold set to median yield in dataset.
    """
    recs = []
    if yield_threshold is None:
        yield_threshold = float(y.median())
    # Disease-based advices (basic rule-set; expand as needed)
    if disease_detected:
        recs.append("Disease detected — recommended actions:")
        recs.append("- Isolate affected plants; remove severely infected material.")
        recs.append("- Apply appropriate fungicide/bactericide/insecticide as per local agricultural guidelines.")
        recs.append("- Rotate crops and avoid planting same crop next season on same land.")
        recs.append("- Improve ventilation and reduce leaf wetness (drainage/irrigation scheduling).")
        recs.append("- Send sample to extension lab for exact pathogen ID and targeted treatment.")
        # Also suggest yield-saving measures
        recs.append("- Monitor remaining field daily and consider targeted nutrient application to strengthen plants.")
    else:
        # No disease detected: base recommendations on predicted yield
        if predicted_yield < yield_threshold:
            recs.append(f"Predicted yield ({predicted_yield:.2f}) is below threshold ({yield_threshold:.2f}). Recommendations:")
            recs.append("- Conduct soil test: correct pH and nutrient deficiencies (N, P, K).")
            recs.append("- Optimize fertilizer type and application timing (split doses).")
            recs.append("- Improve irrigation scheduling: ensure even moisture during critical growth stages.")
            recs.append("- Check seed quality and planting density; consider resistant/tolerant varieties.")
            recs.append("- Consider foliar micro-nutrients and bio-stimulants if recommended by agronomist.")
        else:
            recs.append(f"Predicted yield ({predicted_yield:.2f}) is satisfactory (>= {yield_threshold:.2f}). Suggested actions:")
            recs.append("- Maintain current fertilization and irrigation practices.")
            recs.append("- Monitor pests and diseases regularly.")
            recs.append("- Implement best harvesting and post-harvest handling to reduce losses.")
    return recs

# -----------------------
# 7. Demo: Use functions on an example
# -----------------------
# Example input: fill with values (change to test different cases)
example_input = {
    "average_rain_fall_mm_per_year": 300,
    "pesticides_tonnes": 2.0,
    "avg_temp": 27,
    "Year": 2024,
    "Area": df['Area'].iloc[0] if 'Area' in df.columns else ""
}

predicted = predict_yield(example_input)
print("\nExample predicted yield (hg/ha):", round(predicted, 3))

# Suppose disease detection is reported externally (we ignore CNN here)
disease_flag = False   # set True to see disease-based recommendations
recs = advise(disease_flag, predicted)
print("\nRecommendations:")
for r in recs:
    print("-", r)

# Save sample predictions for test set for your report
sample_out = X_test.reset_index(drop=True).head(20).copy()
sample_out[TARGET + "_actual"] = y_test.reset_index(drop=True).head(20)
sample_out[TARGET + "_pred"] = model.predict(X_test)[:20]
sample_out.to_csv("assignment1_yield_sample_predictions.csv", index=False)
print("\nSaved sample predictions -> assignment1_yield_sample_predictions.csv")
