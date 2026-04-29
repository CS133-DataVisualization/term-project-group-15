import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

X_train = pd.read_csv("Project Assignments/models/ml_splits/X_train.csv")
X_test = pd.read_csv("Project Assignments/models/ml_splits/X_test.csv")
y_reg_train = pd.read_csv("Project Assignments/models/ml_splits/y_reg_train.csv").values.ravel()
y_reg_test = pd.read_csv("Project Assignments/models/ml_splits/y_reg_test.csv").values.ravel()

print("Data loaded successfully")
print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
print(f"Features ({X_train.shape[1]}): {list(X_train.columns)}")

print("\n========== RANDOM FOREST REGRESSION ==========")
rf_reg = RandomForestRegressor(
    n_estimators=400,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

cv_r2 = cross_val_score(rf_reg, X_train, y_reg_train, cv=5, scoring="r2")
cv_mse = cross_val_score(rf_reg, X_train, y_reg_train, cv=5, scoring="neg_mean_squared_error")
cv_rmse = np.sqrt(-cv_mse)

print(f"CV R^2 scores: {cv_r2.round(4)}  -> avg: {cv_r2.mean():.4f}")
print(f"CV RMSE scores: {cv_rmse.round(2)}  -> avg: {cv_rmse.mean():.2f}")

rf_reg.fit(X_train, y_reg_train)
y_pred = rf_reg.predict(X_test)

test_mse = mean_squared_error(y_reg_test, y_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_reg_test, y_pred)

print(f"\nTEST SET:")
print(f"  MSE: {test_mse:.4f}")
print(f"  RMSE: {test_rmse:.4f}")
print(f"  R^2:  {test_r2:.4f}")

print("\n========== MODEL COMPARISON ==========")
comparison = pd.DataFrame({
    "Model": ["Linear Regression (baseline)", "Random Forest"],
    "MSE":   [70.00, round(test_mse, 4)],
    "RMSE":  [round(np.sqrt(70.00), 2), round(test_rmse, 4)],
    "R^2":   [0.14, round(test_r2, 4)],
})
print(comparison.to_string(index=False))

# 1. Actual vs Predicted
plt.figure(figsize=(7, 5))
plt.scatter(y_reg_test, y_pred, alpha=0.3, s=10, color="forestgreen")
lim = max(y_reg_test.max(), y_pred.max())
plt.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect prediction")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.title("Random Forest — Actual vs Predicted Price")
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
plt.title("Random Forest — Residual Plot")
plt.tight_layout()
plt.show()

# 3. Feature Importance
importances = pd.Series(rf_reg.feature_importances_, index=X_train.columns)
top_features = importances.nlargest(15).sort_values()
plt.figure(figsize=(8, 6))
top_features.plot(kind="barh", color="forestgreen")
plt.title("Random Forest — Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# 4. Prediction Error Distribution
plt.figure(figsize=(7, 5))
plt.hist(residuals, bins=60, color="forestgreen", edgecolor="white")
plt.axvline(0, color="red", linestyle="--", lw=1.5)
plt.xlabel("Prediction Error ($)")
plt.ylabel("Count")
plt.title("Random Forest — Distribution of Prediction Errors")
plt.tight_layout()
plt.show()

# SAMPLE PREDICTIONS
print("\n========== SAMPLE PREDICTIONS (first 10 test rows) ==========")
sample = pd.DataFrame({
    "Actual Price ($)": y_reg_test[:10],
    "RF Predicted ($)": y_pred[:10].round(2),
    "Error ($)": (y_reg_test[:10] - y_pred[:10]).round(2),
})
print(sample.to_string(index=False))