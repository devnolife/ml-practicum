"""
Pertemuan 6: Ensemble Methods - Random Forest & Gradient Boosting
Contoh program lengkap untuk bagging dan boosting algorithms
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score
)
from sklearn.datasets import load_iris, load_diabetes, make_classification

# Settings
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 60)
print("PERTEMUAN 6: Ensemble Methods")
print("=" * 60)

# ============================================================================
# 1. RANDOM FOREST CLASSIFICATION
# ============================================================================
print("\n1. RANDOM FOREST CLASSIFICATION")
print("-" * 60)

# Load Iris dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

print(f"Dataset: {X_iris.shape[0]} samples, {X_iris.shape[1]} features")
print(f"Classes: {iris.target_names}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42
)

# Train Single Decision Tree (untuk perbandingan)
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)
y_pred_dt = dt_clf.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

# Train Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"\nSingle Decision Tree Accuracy: {accuracy_dt:.4f}")
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")
print(f"Improvement: {(accuracy_rf - accuracy_dt)*100:.2f}%")

# Cross-validation scores
cv_scores = cross_val_score(rf_clf, X_iris, y_iris, cv=5)
print(f"\nCross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Classification Report
print("\n--- Random Forest Classification Report ---")
print(classification_report(y_test, y_pred_rf, target_names=iris.target_names))

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': iris.feature_names,
    'Importance': rf_clf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n--- Feature Importance ---")
print(feature_importance)

# Visualisasi
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Random Forest Classification - Iris Dataset', fontsize=16, fontweight='bold')

# Plot 1: Confusion Matrix
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=iris.target_names, yticklabels=iris.target_names)
axes[0, 0].set_title(f'Confusion Matrix (Acc: {accuracy_rf:.4f})')
axes[0, 0].set_ylabel('Actual')
axes[0, 0].set_xlabel('Predicted')

# Plot 2: Feature Importance
axes[0, 1].barh(feature_importance['Feature'], feature_importance['Importance'], 
                color='skyblue', edgecolor='black')
axes[0, 1].set_xlabel('Importance')
axes[0, 1].set_title('Feature Importance')
axes[0, 1].invert_yaxis()
axes[0, 1].grid(True, alpha=0.3, axis='x')

# Plot 3: Decision Tree vs Random Forest Comparison
models_comparison = pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest'],
    'Accuracy': [accuracy_dt, accuracy_rf]
})
axes[1, 0].bar(models_comparison['Model'], models_comparison['Accuracy'], 
               color=['#FF6B6B', '#4ECDC4'], edgecolor='black')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].set_title('Model Comparison')
axes[1, 0].set_ylim([0, 1.1])
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Cross-Validation Scores
axes[1, 1].plot(range(1, len(cv_scores)+1), cv_scores, marker='o', 
                linewidth=2, markersize=10, color='green')
axes[1, 1].axhline(y=cv_scores.mean(), color='red', linestyle='--', 
                   label=f'Mean: {cv_scores.mean():.4f}')
axes[1, 1].set_xlabel('Fold')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_title('Cross-Validation Scores')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan06_random_forest_classification.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualisasi Random Forest Classification disimpan")
plt.close()

# ============================================================================
# 2. RANDOM FOREST REGRESSION
# ============================================================================
print("\n2. RANDOM FOREST REGRESSION")
print("-" * 60)

# Load Diabetes dataset
diabetes = load_diabetes()
X_diabetes = diabetes.data
y_diabetes = diabetes.target

print(f"Diabetes Dataset: {X_diabetes.shape[0]} samples, {X_diabetes.shape[1]} features")

# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_diabetes, y_diabetes, test_size=0.3, random_state=42
)

# Train Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_reg, y_train_reg)

# Predict
y_pred_reg = rf_reg.predict(X_test_reg)

# Evaluasi
r2 = r2_score(y_test_reg, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))

print(f"\nR² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# Feature Importance
feature_importance_reg = pd.DataFrame({
    'Feature': diabetes.feature_names,
    'Importance': rf_reg.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n--- Feature Importance (Top 5) ---")
print(feature_importance_reg.head())

# Visualisasi
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Random Forest Regression - Diabetes Dataset', fontsize=16, fontweight='bold')

# Plot 1: Actual vs Predicted
axes[0, 0].scatter(y_test_reg, y_pred_reg, alpha=0.6, color='blue', edgecolors='black')
axes[0, 0].plot([y_test_reg.min(), y_test_reg.max()], 
                [y_test_reg.min(), y_test_reg.max()], 
                'r--', lw=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Values')
axes[0, 0].set_ylabel('Predicted Values')
axes[0, 0].set_title(f'Actual vs Predicted (R² = {r2:.4f})')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Residuals
residuals = y_test_reg - y_pred_reg
axes[0, 1].scatter(y_pred_reg, residuals, alpha=0.6, color='purple', edgecolors='black')
axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Predicted Values')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residual Plot')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Feature Importance
top_features = feature_importance_reg.head(10)
axes[1, 0].barh(top_features['Feature'], top_features['Importance'], 
                color='lightgreen', edgecolor='black')
axes[1, 0].set_xlabel('Importance')
axes[1, 0].set_title('Feature Importance (Top 10)')
axes[1, 0].invert_yaxis()
axes[1, 0].grid(True, alpha=0.3, axis='x')

# Plot 4: Prediction Error Distribution
axes[1, 1].hist(residuals, bins=30, color='salmon', edgecolor='black', alpha=0.7)
axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Prediction Error')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Distribution of Prediction Errors')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan06_random_forest_regression.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualisasi Random Forest Regression disimpan")
plt.close()

# ============================================================================
# 3. GRADIENT BOOSTING CLASSIFICATION
# ============================================================================
print("\n3. GRADIENT BOOSTING CLASSIFICATION")
print("-" * 60)

# Generate synthetic dataset
X_gb, y_gb = make_classification(n_samples=1000, n_features=20, 
                                 n_informative=15, n_redundant=5,
                                 random_state=42)

print(f"Dataset: {X_gb.shape[0]} samples, {X_gb.shape[1]} features")

# Split data
X_train_gb, X_test_gb, y_train_gb, y_test_gb = train_test_split(
    X_gb, y_gb, test_size=0.3, random_state=42
)

# Train Random Forest (untuk perbandingan)
rf_gb = RandomForestClassifier(n_estimators=100, random_state=42)
rf_gb.fit(X_train_gb, y_train_gb)
y_pred_rf_gb = rf_gb.predict(X_test_gb)
accuracy_rf_gb = accuracy_score(y_test_gb, y_pred_rf_gb)

# Train Gradient Boosting
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                    max_depth=3, random_state=42)
gb_clf.fit(X_train_gb, y_train_gb)
y_pred_gb = gb_clf.predict(X_test_gb)
accuracy_gb = accuracy_score(y_test_gb, y_pred_gb)

print(f"\nRandom Forest Accuracy: {accuracy_rf_gb:.4f}")
print(f"Gradient Boosting Accuracy: {accuracy_gb:.4f}")

# Classification Reports
print("\n--- Random Forest Report ---")
print(classification_report(y_test_gb, y_pred_rf_gb))

print("\n--- Gradient Boosting Report ---")
print(classification_report(y_test_gb, y_pred_gb))

# Visualisasi
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Gradient Boosting vs Random Forest', fontsize=16, fontweight='bold')

# Plot 1: Confusion Matrix - Random Forest
cm_rf = confusion_matrix(y_test_gb, y_pred_rf_gb)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title(f'Random Forest (Acc: {accuracy_rf_gb:.4f})')
axes[0, 0].set_ylabel('Actual')
axes[0, 0].set_xlabel('Predicted')

# Plot 2: Confusion Matrix - Gradient Boosting
cm_gb = confusion_matrix(y_test_gb, y_pred_gb)
sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Greens', ax=axes[0, 1])
axes[0, 1].set_title(f'Gradient Boosting (Acc: {accuracy_gb:.4f})')
axes[0, 1].set_ylabel('Actual')
axes[0, 1].set_xlabel('Predicted')

# Plot 3: Feature Importance Comparison
top_n = 10
fi_rf = pd.DataFrame({
    'Feature': [f'F{i}' for i in range(X_gb.shape[1])],
    'Importance': rf_gb.feature_importances_
}).sort_values('Importance', ascending=False).head(top_n)

fi_gb = pd.DataFrame({
    'Feature': [f'F{i}' for i in range(X_gb.shape[1])],
    'Importance': gb_clf.feature_importances_
}).sort_values('Importance', ascending=False).head(top_n)

x = np.arange(len(fi_rf))
width = 0.35

axes[1, 0].barh(x, fi_rf['Importance'].values, width, label='Random Forest', color='skyblue')
axes[1, 0].barh(x + width, fi_gb['Importance'].values, width, label='Gradient Boosting', color='lightgreen')
axes[1, 0].set_yticks(x + width / 2)
axes[1, 0].set_yticklabels(fi_rf['Feature'])
axes[1, 0].invert_yaxis()
axes[1, 0].set_xlabel('Importance')
axes[1, 0].set_title(f'Feature Importance (Top {top_n})')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='x')

# Plot 4: Learning Curve (Gradient Boosting)
# Training error over iterations
train_scores = []
test_scores = []
for i, y_pred_train in enumerate(gb_clf.staged_predict(X_train_gb)):
    train_scores.append(accuracy_score(y_train_gb, y_pred_train))
    
for i, y_pred_test in enumerate(gb_clf.staged_predict(X_test_gb)):
    test_scores.append(accuracy_score(y_test_gb, y_pred_test))

axes[1, 1].plot(range(1, len(train_scores)+1), train_scores, label='Training', linewidth=2)
axes[1, 1].plot(range(1, len(test_scores)+1), test_scores, label='Testing', linewidth=2)
axes[1, 1].set_xlabel('Number of Estimators')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_title('Gradient Boosting Learning Curve')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan06_gradient_boosting.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualisasi Gradient Boosting disimpan")
plt.close()

# ============================================================================
# 4. HYPERPARAMETER TUNING
# ============================================================================
print("\n4. HYPERPARAMETER TUNING - Random Forest")
print("-" * 60)

# Parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

print("Parameter Grid:")
for key, value in param_grid.items():
    print(f"  {key}: {value}")

print("\nMelakukan Grid Search...")

# GridSearchCV
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train_gb, y_train_gb)

print("\n✓ Grid Search selesai!")

# Best parameters
print("\nBest Parameters:")
for key, value in grid_search.best_params_.items():
    print(f"  {key}: {value}")

print(f"\nBest CV Score: {grid_search.best_score_:.4f}")

# Test with best model
best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test_gb)
accuracy_best = accuracy_score(y_test_gb, y_pred_best)

print(f"Test Accuracy (Best Model): {accuracy_best:.4f}")
print(f"Improvement from default: {(accuracy_best - accuracy_rf_gb)*100:.2f}%")

# Visualisasi Grid Search Results
results_df = pd.DataFrame(grid_search.cv_results_)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Hyperparameter Tuning Results', fontsize=16, fontweight='bold')

# Plot 1: n_estimators vs accuracy
grouped = results_df.groupby('param_n_estimators')['mean_test_score'].mean()
axes[0].plot(grouped.index, grouped.values, marker='o', linewidth=2, markersize=10)
axes[0].set_xlabel('Number of Estimators')
axes[0].set_ylabel('Mean CV Accuracy')
axes[0].set_title('Effect of n_estimators')
axes[0].grid(True, alpha=0.3)

# Plot 2: max_depth vs accuracy
grouped_depth = results_df.groupby('param_max_depth')['mean_test_score'].mean().sort_index()
axes[1].bar(range(len(grouped_depth)), grouped_depth.values, color='lightgreen', edgecolor='black')
axes[1].set_xticks(range(len(grouped_depth)))
axes[1].set_xticklabels([str(x) for x in grouped_depth.index])
axes[1].set_xlabel('Max Depth')
axes[1].set_ylabel('Mean CV Accuracy')
axes[1].set_title('Effect of max_depth')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan06_hyperparameter_tuning.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualisasi Hyperparameter Tuning disimpan")
plt.close()

print("\n" + "=" * 60)
print("SELESAI - Pertemuan 6")
print("=" * 60)
print("\nRingkasan yang dipelajari:")
print("✓ Random Forest Classification")
print("✓ Random Forest Regression")
print("✓ Gradient Boosting Classification")
print("✓ Feature Importance Analysis")
print("✓ Model Comparison (Bagging vs Boosting)")
print("✓ Hyperparameter Tuning dengan GridSearchCV")
print("✓ Cross-Validation")
print("\nFile yang dibuat:")
print("- pertemuan06_random_forest_classification.png")
print("- pertemuan06_random_forest_regression.png")
print("- pertemuan06_gradient_boosting.png")
print("- pertemuan06_hyperparameter_tuning.png")
