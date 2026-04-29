import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import cross_val_score

# LOAD DATA
X_train = pd.read_csv("ml_splits/X_train.csv")
X_test = pd.read_csv("ml_splits/X_test.csv")
y_reg_train = pd.read_csv("ml_splits/y_reg_train.csv").values.ravel()
y_reg_test = pd.read_csv("ml_splits/y_reg_test.csv").values.ravel()
y_cls_train = pd.read_csv("ml_splits/y_cls_train.csv").values.ravel()
y_cls_test = pd.read_csv("ml_splits/y_cls_test.csv").values.ravel()

print("Data loaded successfully")
print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
print(f"Features ({X_train.shape[1]}): {list(X_train.columns)}")

print(f"Regression target -> min: {y_reg_train.min()}, max: {y_reg_train.max()}, mean: {y_reg_train.mean():.2f}")
print(f"Classification target -> unique values: {np.unique(y_cls_train, return_counts=True)}")

has_both_classes = len(np.unique(y_cls_train)) == 2
if not has_both_classes:
    print("\nWARNING: y_cls_train only has one class. Classification will be skipped.")
else:
    print("Both Budget and Premium classes are present.")

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

# Cross-validation
cv_r2_reg = cross_val_score(rf_reg, X_train, y_reg_train, cv=5, scoring="r2")
cv_mse_reg = cross_val_score(rf_reg, X_train, y_reg_train, cv=5, scoring="neg_mean_squared_error")
cv_rmse_reg = np.sqrt(-cv_mse_reg)

print(f"CV R^2   scores: {cv_r2_reg.round(4)}  -> avg: {cv_r2_reg.mean():.4f}")
print(f"CV RMSE scores: {cv_rmse_reg.round(2)}  -> avg: {cv_rmse_reg.mean():.2f}")

# Train + predict
rf_reg.fit(X_train, y_reg_train)
y_pred_reg = rf_reg.predict(X_test)

# Test metrics
test_mse_reg = mean_squared_error(y_reg_test, y_pred_reg)
test_rmse_reg = np.sqrt(test_mse_reg)
test_r2_reg = r2_score(y_reg_test, y_pred_reg)

print(f"\nTEST SET:")
print(f"  MSE:  {test_mse_reg:.4f}")
print(f"  RMSE: {test_rmse_reg:.4f}  <- avg prediction error in dollars")
print(f"  R^2:  {test_r2_reg:.4f}")

print("\n========== RANDOM FOREST CLASSIFICATION ==========")

if not has_both_classes:
    print("Skipping classification because only one class exists in y_cls_train.")
else:
    class_counts = np.bincount(y_cls_train.astype(int))
    print(f"Class balance -> Budget: {class_counts[0]} | Premium: {class_counts[1]}")

    rf_cls = RandomForestClassifier(
        n_estimators=400,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # Cross-validation
    cv_acc_cls = cross_val_score(rf_cls, X_train, y_cls_train, cv=5, scoring="accuracy")
    cv_f1_cls = cross_val_score(rf_cls, X_train, y_cls_train, cv=5, scoring="f1")

    print(f"CV Accuracy scores: {cv_acc_cls.round(4)}  -> avg: {cv_acc_cls.mean():.4f}")
    print(f"CV F1 scores:       {cv_f1_cls.round(4)}  -> avg: {cv_f1_cls.mean():.4f}")

    # Train + predict
    rf_cls.fit(X_train, y_cls_train)
    y_pred_cls = rf_cls.predict(X_test)

    # Test metrics
    test_acc_cls = accuracy_score(y_cls_test, y_pred_cls)
    test_prec_cls = precision_score(y_cls_test, y_pred_cls, zero_division=0)
    test_rec_cls = recall_score(y_cls_test, y_pred_cls, zero_division=0)
    test_f1_cls = f1_score(y_cls_test, y_pred_cls, zero_division=0)

    print(f"\nTEST SET:")
    print(f"  Accuracy:  {test_acc_cls:.4f}")
    print(f"  Precision: {test_prec_cls:.4f}")
    print(f"  Recall:    {test_rec_cls:.4f}")
    print(f"  F1-Score:  {test_f1_cls:.4f}")

print("\n========== REGRESSION MODEL COMPARISON ==========")
comparison_reg = pd.DataFrame({
    "Model": ["Linear Regression (baseline)", "Random Forest"],
    "MSE": [70.00, round(test_mse_reg, 4)],
    "RMSE": [round(np.sqrt(70.00), 2), round(test_rmse_reg, 4)],
    "R^2": [0.14, round(test_r2_reg, 4)],
})
print(comparison_reg.to_string(index=False))

if has_both_classes:
    print("\n========== CLASSIFICATION MODEL RESULTS ==========")
    comparison_cls = pd.DataFrame({
        "Model": ["Random Forest Classifier"],
        "Accuracy": [round(test_acc_cls, 4)],
        "Precision": [round(test_prec_cls, 4)],
        "Recall": [round(test_rec_cls, 4)],
        "F1": [round(test_f1_cls, 4)],
    })
    print(comparison_cls.to_string(index=False))

# 1. Actual vs Predicted (Regression)
plt.figure(figsize=(7, 5))
plt.scatter(y_reg_test, y_pred_reg, alpha=0.3, s=10, color="forestgreen")
lim = max(y_reg_test.max(), y_pred_reg.max())
plt.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect prediction")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.title("Random Forest Regression — Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.show()

# 2. Residual Plot (Regression)
residuals = y_reg_test - y_pred_reg
plt.figure(figsize=(7, 5))
plt.scatter(y_pred_reg, residuals, alpha=0.3, s=10, color="darkorange")
plt.axhline(0, color="red", linestyle="--", lw=1.5)
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.title("Random Forest Regression — Residual Plot")
plt.tight_layout()
plt.show()

# 3. Prediction Error Distribution (Regression)
plt.figure(figsize=(7, 5))
plt.hist(residuals, bins=60, color="forestgreen", edgecolor="white")
plt.axvline(0, color="red", linestyle="--", lw=1.5)
plt.xlabel("Prediction Error ($)")
plt.ylabel("Count")
plt.title("Random Forest Regression — Distribution of Prediction Errors")
plt.tight_layout()
plt.show()

# 4. Feature Importance (Regression)
importances_reg = pd.Series(rf_reg.feature_importances_, index=X_train.columns)
top15_reg = importances_reg.nlargest(15).sort_values()
plt.figure(figsize=(8, 6))
top15_reg.plot(kind="barh", color="forestgreen")
plt.title("Random Forest — Top 15 Feature Importances (Regression)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# 5. Confusion Matrix (Classification)
if has_both_classes:
    cm = confusion_matrix(y_cls_test, y_pred_cls, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Budget", "Premium"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Random Forest — Confusion Matrix (Budget vs Premium)")
    plt.tight_layout()
    plt.show()

# 6. Feature Importance (Classification)
if has_both_classes:
    importances_cls = pd.Series(rf_cls.feature_importances_, index=X_train.columns)
    top15_cls = importances_cls.nlargest(15).sort_values()
    plt.figure(figsize=(8, 6))
    top15_cls.plot(kind="barh", color="seagreen")
    plt.title("Random Forest — Top 15 Feature Importances (Classification)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()

print("\n========== SAMPLE PREDICTIONS (first 10 test rows) ==========")

sample = pd.DataFrame({
    "Actual Price ($)": y_reg_test[:10],
    "RF Predicted ($)": y_pred_reg[:10].round(2),
    "Error ($)": (y_reg_test[:10] - y_pred_reg[:10]).round(2),
})

if has_both_classes:
    sample["Actual Label"] = ["Premium" if v == 1 else "Budget" for v in y_cls_test[:10]]
    sample["RF Label Pred"] = ["Premium" if v == 1 else "Budget" for v in y_pred_cls[:10]]

print(sample.to_string(index=False))