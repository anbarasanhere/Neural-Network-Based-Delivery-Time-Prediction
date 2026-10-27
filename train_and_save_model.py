"""
Training script to extract and save model artifacts from the notebook.
Run this once to generate model.pkl, scaler.pkl, and feature_columns.json
"""
import pandas as pd
import numpy as np
import json
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("Loading data...")
# Download data if not present
if not os.path.exists('data_2.csv'):
    import gdown
    gdown.download('https://drive.google.com/uc?id=1IpoNhNKlPABp_KV16qP4FmgE-WhrfZGd', 'data_2.csv', quiet=False)

df = pd.read_csv('data_2.csv')
print(f"Data shape: {df.shape}")

# ============ PREPROCESSING (from notebook) ============
# Handle Datetime Columns
df['created_at'] = pd.to_datetime(df['created_at'])
df['actual_delivery_time'] = pd.to_datetime(df['actual_delivery_time'])

# Calculate delivery time in minutes
df['delivery_time'] = (df['actual_delivery_time'] - df['created_at']).dt.total_seconds() / 60

# Encode categorical columns
df = pd.get_dummies(df, columns=['store_primary_category', 'order_protocol'], drop_first=True)

# Handle negative/invalid delivery times
df = df[df['delivery_time'] > 0]

# Log transform target
df['delivery_time'] = np.log1p(df['delivery_time'])

# Remove outliers using IQR (5th and 95th percentile as in notebook)
Q1 = df['delivery_time'].quantile(0.05)
Q3 = df['delivery_time'].quantile(0.95)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['delivery_time'] >= lower_bound) & (df['delivery_time'] <= upper_bound)]

# Define features and target
X = df.drop('delivery_time', axis=1)
y = df['delivery_time']

# Remove non-numeric columns
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_cols:
    X = X.drop(columns=non_numeric_cols)

# Remove constant features
constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
if constant_cols:
    X = X.drop(columns=constant_cols)

# Handle missing values with KNN Imputer
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training shape: {X_train_scaled.shape}")
print(f"Test shape: {X_test_scaled.shape}")

# ============ MODEL ARCHITECTURE (from notebook) ============
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)  # Linear output for regression
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train
print("Training model...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate
test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest MAE (log scale): {test_mae:.4f}")
print(f"Test MSE (log scale): {test_loss:.4f}")

# Convert back to minutes for interpretability
y_pred_log = model.predict(X_test_scaled).flatten()
y_pred = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test.values)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae_min = mean_absolute_error(y_test_actual, y_pred)
rmse_min = np.sqrt(mean_squared_error(y_test_actual, y_pred))
r2 = r2_score(y_test_actual, y_pred)
print(f"Test MAE (minutes): {mae_min:.2f}")
print(f"Test RMSE (minutes): {rmse_min:.2f}")
print(f"Test R²: {r2:.4f}")

# ============ SAVE ARTIFACTS ============
artifacts_dir = 'model_artifacts'
os.makedirs(artifacts_dir, exist_ok=True)

# Save model
model.save(os.path.join(artifacts_dir, 'delivery_time_model.keras'))
print(f"Model saved to {artifacts_dir}/delivery_time_model.keras")

# Save scaler
with open(os.path.join(artifacts_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler saved to {artifacts_dir}/scaler.pkl")

# Save imputer
with open(os.path.join(artifacts_dir, 'imputer.pkl'), 'wb') as f:
    pickle.dump(imputer, f)
print(f"Imputer saved to {artifacts_dir}/imputer.pkl")

# Save feature columns
feature_columns = X.columns.tolist()
with open(os.path.join(artifacts_dir, 'feature_columns.json'), 'w') as f:
    json.dump(feature_columns, f)
print(f"Feature columns saved to {artifacts_dir}/feature_columns.json")

# Save categorical column info for UI
categorical_info = {
    'store_primary_category': sorted([c.replace('store_primary_category_', '') for c in feature_columns if c.startswith('store_primary_category_')]),
    'order_protocol': sorted([c.replace('order_protocol_', '') for c in feature_columns if c.startswith('order_protocol_')])
}
with open(os.path.join(artifacts_dir, 'categorical_info.json'), 'w') as f:
    json.dump(categorical_info, f)
print(f"Categorical info saved to {artifacts_dir}/categorical_info.json")

# Save training metrics
metrics = {
    'test_mae_log': float(test_mae),
    'test_mse_log': float(test_loss),
    'test_mae_minutes': float(mae_min),
    'test_rmse_minutes': float(rmse_min),
    'test_r2': float(r2),
    'n_features': len(feature_columns),
    'n_train_samples': len(X_train),
    'n_test_samples': len(X_test)
}
with open(os.path.join(artifacts_dir, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved to {artifacts_dir}/metrics.json")

print("\n✅ All artifacts saved successfully!")