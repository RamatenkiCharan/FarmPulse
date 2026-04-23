"""
train_model.py — Pharma Manufacturing Model Trainer
=====================================================
Generates synthetic pharmaceutical batch sensor data, engineers features,
trains a MultiOutputRegressor(XGBRegressor) for predicting Energy,
Vibration_Target, and Quality_Metric, and saves model + scaler as .joblib.

Usage:
    python train_model.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import joblib

# ─── Configuration ───────────────────────────────────────────────────────────
NUM_BATCHES = 200          # Total batches to simulate for training
READINGS_PER_BATCH = 100   # Sensor readings per batch
RANDOM_SEED = 42
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

np.random.seed(RANDOM_SEED)


# ─── 1. Simulate Raw Sensor Data ─────────────────────────────────────────────
def simulate_batch_readings(batch_id: str, n_readings: int = READINGS_PER_BATCH):
    """
    Simulate one batch of sensor readings for:
      Temperature (°C), Pressure (psi), Power (kW), Vibration (mm/s)
    """
    return pd.DataFrame({
        "Batch_ID": batch_id,
        "Temperature": np.random.normal(loc=75, scale=5, size=n_readings),
        "Pressure":    np.random.normal(loc=30, scale=3, size=n_readings),
        "Power":       np.random.normal(loc=50, scale=8, size=n_readings),
        "Vibration":   np.random.normal(loc=2.5, scale=0.5, size=n_readings),
    })


# ─── 2. Aggregate Sensor Data ────────────────────────────────────────────────
def aggregate_batch(df: pd.DataFrame) -> pd.Series:
    """Compute mean, max, std for each sensor column in a single batch."""
    sensors = ["Temperature", "Pressure", "Power", "Vibration"]
    agg = {}
    agg["Batch_ID"] = df["Batch_ID"].iloc[0]
    for s in sensors:
        agg[f"{s}_mean"] = df[s].mean()
        agg[f"{s}_max"]  = df[s].max()
        agg[f"{s}_std"]  = df[s].std()
    return pd.Series(agg)


# ─── 3. Build Full Training Dataset ──────────────────────────────────────────
print("🔬 Simulating sensor data for", NUM_BATCHES, "batches …")

batches = []
for i in range(1, NUM_BATCHES + 1):
    bid = f"T{i:03d}"
    raw = simulate_batch_readings(bid)
    batches.append(aggregate_batch(raw))

agg_df = pd.DataFrame(batches)

# Synthetic Summary (Pass/Fail) and Operator assignments
agg_df["Status"]   = np.random.choice(["Pass", "Fail"], size=NUM_BATCHES, p=[0.85, 0.15])
agg_df["Operator"] = np.random.choice(["Op_A", "Op_B", "Op_C", "Op_D"], size=NUM_BATCHES)

# Feature engineering
agg_df["Energy_Pattern_Ratio"] = agg_df["Power_mean"] / agg_df["Pressure_mean"]

# One-hot encode categoricals
agg_df = pd.get_dummies(agg_df, columns=["Status", "Operator"], dtype=float)

# Fill any NaN
agg_df.fillna(0, inplace=True)


# ─── 4. Prepare Features & Targets ───────────────────────────────────────────
# Synthetic targets with realistic relationships to sensor data
agg_df["Energy"]           = (agg_df["Power_mean"] * 1.2
                              + agg_df["Temperature_mean"] * 0.3
                              + np.random.normal(0, 2, NUM_BATCHES))

agg_df["Vibration_Target"] = (agg_df["Vibration_mean"] * 1.5
                              + agg_df["Pressure_std"] * 0.4
                              + np.random.normal(0, 0.3, NUM_BATCHES))

agg_df["Quality_Metric"]   = (100
                              - agg_df["Vibration_mean"] * 3
                              - agg_df["Temperature_std"] * 0.5
                              + agg_df["Energy_Pattern_Ratio"] * 2
                              + np.random.normal(0, 1.5, NUM_BATCHES))

target_cols  = ["Energy", "Vibration_Target", "Quality_Metric"]
feature_cols = [c for c in agg_df.columns if c not in target_cols + ["Batch_ID"]]

X = agg_df[feature_cols].values
y = agg_df[target_cols].values

print(f"✅ Feature matrix shape: {X.shape}  |  Target shape: {y.shape}")
print(f"   Features: {feature_cols}")


# ─── 5. Scale & Train ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

model = MultiOutputRegressor(
    XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_SEED,
        verbosity=0,
    )
)

print("🏋️  Training MultiOutputRegressor(XGBRegressor) …")
model.fit(X_train_sc, y_train)

# Quick evaluation
from sklearn.metrics import r2_score
y_pred = model.predict(X_test_sc)
for i, name in enumerate(target_cols):
    r2 = r2_score(y_test[:, i], y_pred[:, i])
    print(f"   {name:20s}  R² = {r2:.4f}")


# ─── 6. Save Artifacts ───────────────────────────────────────────────────────
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

model_path  = os.path.join(ARTIFACTS_DIR, "multi_target_model.joblib")
scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.joblib")
meta_path   = os.path.join(ARTIFACTS_DIR, "feature_columns.joblib")

joblib.dump(model,        model_path)
joblib.dump(scaler,       scaler_path)
joblib.dump(feature_cols, meta_path)

print(f"\n💾 Saved:")
print(f"   Model  → {model_path}")
print(f"   Scaler → {scaler_path}")
print(f"   Cols   → {meta_path}")
print("\n🎉 Done! You can now run:  streamlit run app.py")
