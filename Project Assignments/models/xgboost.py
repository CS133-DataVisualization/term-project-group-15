import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

# LOAD DATA
X_train     = pd.read_csv("Project Assignments/models/ml_splits/X_train.csv")
X_test      = pd.read_csv("Project Assignments/models/ml_splits/X_test.csv")
y_reg_train = pd.read_csv("Project Assignments/models/ml_splits/y_reg_train.csv").values.ravel()
y_reg_test  = pd.read_csv("Project Assignments/models/ml_splits/y_reg_test.csv").values.ravel()

print("✅ Data loaded successfully")
print(f"   Train rows: {len(X_train)} | Test rows: {len(X_test)}")
print(f"   Features ({X_train.shape[1]}): {list(X_train.columns)}")

# XGBOOST REGRESSION
print("\n========== XGBOOST REGRESSION ==========")

xgb_reg = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)

# 5-fold cross-validation on training data
cv_r2  = cross_val_score(xgb_reg, X_train, y_reg_train, cv=5, scoring="r2")
cv_mse = cross_val_score(xgb_reg, X_train, y_reg_train, cv=5, scoring="neg_mean_squared_error")
cv_rmse = np.sqrt(-cv_mse)

print(f"CV R²   scores: {cv_r2.round(4)}  →  avg: {cv_r2.mean():.4f}")
print(f"CV RMSE scores: {cv_rmse.round(2)}  →  avg: {cv_rmse.mean():.2f}")

# Train on full training set
xgb_reg.fit(X_train, y_reg_train)
y_pred = xgb_reg.predict(X_test)

test_mse  = mean_squared_error(y_reg_test, y_pred)
test_rmse = np.sqrt(test_mse)
test_r2   = r2_score(y_reg_test, y_pred)

print(f"\nTEST SET:")
print(f"  MSE:  {test_mse:.4f}")
print(f"  RMSE: {test_rmse:.4f}  ← avg prediction error in dollars")
print(f"  R²:   {test_r2:.4f}")

# MODEL COMPARISON (with linear regression)
print("\n========== MODEL COMPARISON ==========")
comparison = pd.DataFrame({
    "Model": ["Linear Regression (baseline)", "XGBoost"],
    "MSE":   [70.00, round(test_mse, 4)],
    "RMSE":  [round(np.sqrt(70.00), 2), round(test_rmse, 4)],
    "R²":    [0.14,  round(test_r2, 4)],
})
print(comparison.to_string(index=False))

# VISUALIZATIONS

# 1. Actual vs Predicted
plt.figure(figsize=(7, 5))
plt.scatter(y_reg_test, y_pred, alpha=0.3, s=10, color="steelblue")
lim = max(y_reg_test.max(), y_pred.max())
plt.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect prediction")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.title("XGBoost — Actual vs Predicted Price")
plt.legend()
plt.tight_layout()
plt.show()

# 2. Residual Plot
residuals = y_reg_test - y_pred
plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals, alpha=0.3, s=10, color="darkorange")
plt.axhline(0, color="red", linestyle="--", lw=1.5)
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.title("XGBoost — Residual Plot")
plt.tight_layout()
plt.show()

# 3. Feature Importance
importances = pd.Series(xgb_reg.feature_importances_, index=X_train.columns)
top_features = importances.nlargest(15).sort_values()
plt.figure(figsize=(8, 6))
top_features.plot(kind="barh", color="steelblue")
plt.title("XGBoost — Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# 4. Prediction Error Distribution
plt.figure(figsize=(7, 5))
plt.hist(residuals, bins=60, color="steelblue", edgecolor="white")
plt.axvline(0, color="red", linestyle="--", lw=1.5)
plt.xlabel("Prediction Error ($)")
plt.ylabel("Count")
plt.title("XGBoost — Distribution of Prediction Errors")
plt.tight_layout()
plt.show()

# SAMPLE PREDICTIONS
print("\n========== SAMPLE PREDICTIONS (first 10 test rows) ==========")
sample = pd.DataFrame({
    "Actual Price ($)":  y_reg_test[:10],
    "XGB Predicted ($)":     y_pred[:10].round(2),
    "Error ($)":         (y_reg_test[:10] - y_pred[:10]).round(2),
})
print(sample.to_string(index=False))