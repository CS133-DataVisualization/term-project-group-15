import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# Load the train/test splits 
X_train = pd.read_csv("Project Assignments/models/ml_splits/X_train.csv")
X_test = pd.read_csv("Project Assignments/models/ml_splits/X_test.csv")
y_train = pd.read_csv("Project Assignments/models/ml_splits/y_reg_train.csv").values.ravel()
y_test = pd.read_csv("Project Assignments/models/ml_splits/y_reg_test.csv").values.ravel()

# Cross-validation (no scaling)
model = LinearRegression()

cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

print("Cross-validation R^2 scores:", cv_scores)
print("Average CV R^2:", cv_scores.mean())

# Train/evaluate the model without scaling
model_no_scale = LinearRegression()
model_no_scale.fit(X_train, y_train)

y_pred_no_scale = model_no_scale.predict(X_test)

mse_no_scale = mean_squared_error(y_test, y_pred_no_scale)
r2_no_scale = r2_score(y_test, y_pred_no_scale)

print("WITHOUT SCALING")
print("MSE:", mse_no_scale)
print("R^2:", r2_no_scale)

# Apply scaling and repeat the process
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

cv_scores_scaled = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')

print("\nScaled Cross-validation R^2 scores:", cv_scores_scaled)
print("Scaled Average CV R^2:", cv_scores_scaled.mean())

# Train/evaluate the model with scaling
model_scaled = LinearRegression()
model_scaled.fit(X_train_scaled, y_train)

y_pred_scaled = model_scaled.predict(X_test_scaled)

mse_scaled = mean_squared_error(y_test, y_pred_scaled)
r2_scaled = r2_score(y_test, y_pred_scaled)

print("\nWITH SCALING")
print("MSE:", mse_scaled)
print("R^2:", r2_scaled)

# Visualizations for the unscaled model
plt.scatter(y_test, y_pred_no_scale, alpha=0.5)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.show()

# Residual plot for the unscaled model
residuals = y_test - y_pred_no_scale

plt.scatter(y_pred_no_scale, residuals, alpha=0.5)
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.axhline(0)
plt.show()