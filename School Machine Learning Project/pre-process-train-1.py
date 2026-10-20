import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load Boston Housing data
df = pd.read_csv("boston_housing.csv")

# Display basic info
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# ===================== Data Preparation =====================
# All features are already numeric - no encoding needed!

# Split features/target
X = df.drop("medv", axis=1)  # All 13 predictors
y = df["medv"]               # Target: median house value

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on train, transform train
X_test_scaled  = scaler.transform(X_test)        # only transform test

print(f"\nScaled training shape: {X_train_scaled.shape}")
print(f"Scaled test shape: {X_test_scaled.shape}")

# ===================== Baseline Model =====================
print("\n" + "="*60)
print("BASELINE MODEL (Mean Prediction)")
print("="*60)

baseline = np.mean(y_train)
y_pred_base = np.full_like(y_test, baseline, dtype=float)

mse_base = mean_squared_error(y_test, y_pred_base)
rmse_base = np.sqrt(mse_base)
mae_base = mean_absolute_error(y_test, y_pred_base)
r2_base = r2_score(y_test, y_pred_base)

print(f"Baseline prediction (mean): ${baseline:.2f}k")
print(f"RMSE: ${rmse_base:.2f}k")
print(f"MAE:  ${mae_base:.2f}k")
print(f"R²:   {r2_base:.4f}")

# ===================== Linear Regression =====================
print("\n" + "="*60)
print("LINEAR REGRESSION")
print("="*60)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"RMSE: ${rmse_lr:.2f}k")
print(f"MAE:  ${mae_lr:.2f}k")
print(f"R²:   {r2_lr:.4f}")

# Most important features (top 5 by absolute coefficient)
feature_importance_lr = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lr.coef_
}).sort_values('coefficient', key=abs, ascending=False)

print("\nTop 5 most influential features:")
print(feature_importance_lr.head())

# ===================== Random Forest Regressor =====================
print("\n" + "="*60)
print("RANDOM FOREST REGRESSOR")
print("="*60)

rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X_train, y_train)     # RF can use unscaled features
y_pred_rf = rf.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"RMSE: ${rmse_rf:.2f}k")
print(f"MAE:  ${mae_rf:.2f}k")
print(f"R²:   {r2_rf:.4f}")

# Feature importance from Random Forest
feature_importance_rf = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 most important features:")
print(feature_importance_rf.head())

# ===================== Model Comparison =====================
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)

results = pd.DataFrame({
    'Model': ['Baseline (Mean)', 'Linear Regression', 'Random Forest'],
    'RMSE': [rmse_base, rmse_lr, rmse_rf],
    'MAE': [mae_base, mae_lr, mae_rf],
    'R²': [r2_base, r2_lr, r2_rf]
})

print(results.to_string(index=False))

# Best model
best_model_idx = results['R²'].idxmax()
best_model = results.loc[best_model_idx, 'Model']
print(f"\n✓ Best performing model: {best_model} (R² = {results.loc[best_model_idx, 'R²']:.4f})")

# Improvement over baseline
improvement_lr = ((rmse_base - rmse_lr) / rmse_base) * 100
improvement_rf = ((rmse_base - rmse_rf) / rmse_base) * 100

print(f"\nImprovement over baseline:")
print(f"  Linear Regression: {improvement_lr:.1f}% reduction in RMSE")
print(f"  Random Forest:     {improvement_rf:.1f}% reduction in RMSE")
