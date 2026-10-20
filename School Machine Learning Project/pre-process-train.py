import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


df = pd.read_csv("insurance.csv")

# 1) Encode categoricals first if needed
df_enc = df.copy()
df_enc["sex"] = df_enc["sex"].map({"female": 0, "male": 1})
df_enc["smoker"] = df_enc["smoker"].map({"no": 0, "yes": 1})
df_enc = pd.get_dummies(df_enc, columns=["region"], drop_first=True)

# 2) Split features/target
X = df_enc.drop("charges", axis=1)
y = df_enc["charges"]

# 3) Train-test split (split BEFORE scaling)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4) Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on train, transform train
X_test_scaled  = scaler.transform(X_test)       # only transform test

print(X_train_scaled.shape, X_test_scaled.shape)



# ===================== Baseline =====================
baseline = np.mean(y_train)
y_pred_base = np.full_like(y_test, baseline, dtype=float)

mse_base = mean_squared_error(y_test, y_pred_base)
rmse_base = mse_base ** 0.5
print("Baseline RMSE:", rmse_base)
print("Baseline MAE:", mean_absolute_error(y_test, y_pred_base))
print("Baseline R2:", r2_score(y_test, y_pred_base))

# ===================== Linear Regression =====================
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = mse_lr ** 0.5
print("LR RMSE:", rmse_lr)
print("LR MAE:", mean_absolute_error(y_test, y_pred_lr))
print("LR R2:", r2_score(y_test, y_pred_lr))

# ===================== Random Forest =====================
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)     # RF can use unscaled features
y_pred_rf = rf.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = mse_rf ** 0.5
print("RF RMSE:", rmse_rf)
print("RF MAE:", mean_absolute_error(y_test, y_pred_rf))
print("RF R2:", r2_score(y_test, y_pred_rf))
