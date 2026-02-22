"""
Pertemuan 4: Classification - Logistic Regression & Decision Tree
Contoh program lengkap untuk Binary dan Multi-class Classification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, make_classification

# Settings
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

print("=" * 60)
print("PERTEMUAN 4: Classification")
print("=" * 60)

# ============================================================================
# 1. BINARY CLASSIFICATION dengan Logistic Regression
# ============================================================================
print("\n1. BINARY CLASSIFICATION - Logistic Regression")
print("-" * 60)

# Generate synthetic dataset
np.random.seed(42)
X_binary, y_binary = make_classification(
    n_samples=300,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

print(f"Dataset: {X_binary.shape[0]} samples, {X_binary.shape[1]} features")
print(f"Class distribution: Class 0 = {(y_binary==0).sum()}, Class 1 = {(y_binary==1).sum()}")

# Split data
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_binary, y_binary, test_size=0.3, random_state=42
)

# Scaling (penting untuk Logistic Regression!)
scaler_bin = StandardScaler()
X_train_bin_scaled = scaler_bin.fit_transform(X_train_bin)
X_test_bin_scaled = scaler_bin.transform(X_test_bin)

# Train model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_bin_scaled, y_train_bin)

# Predict
y_pred_bin = log_reg.predict(X_test_bin_scaled)
y_pred_proba_bin = log_reg.predict_proba(X_test_bin_scaled)[:, 1]

# Evaluasi
print("\n--- Evaluasi Model ---")
print(f"Accuracy: {accuracy_score(y_test_bin, y_pred_bin):.4f}")
print(f"Precision: {precision_score(y_test_bin, y_pred_bin):.4f}")
print(f"Recall: {recall_score(y_test_bin, y_pred_bin):.4f}")
print(f"F1-Score: {f1_score(y_test_bin, y_pred_bin):.4f}")

# Confusion Matrix
print("\n--- Confusion Matrix ---")
cm_bin = confusion_matrix(y_test_bin, y_pred_bin)
print(cm_bin)
print(f"\nTrue Negative (TN): {cm_bin[0,0]}")
print(f"False Positive (FP): {cm_bin[0,1]}")
print(f"False Negative (FN): {cm_bin[1,0]}")
print(f"True Positive (TP): {cm_bin[1,1]}")

# Visualisasi Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Binary Classification - Logistic Regression', fontsize=16, fontweight='bold')

# Plot 1: Confusion Matrix
sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
            xticklabels=['Class 0', 'Class 1'], 
            yticklabels=['Class 0', 'Class 1'])
axes[0].set_title('Confusion Matrix')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# Plot 2: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test_bin, y_pred_proba_bin)
roc_auc = auc(fpr, tpr)

axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve')
axes[1].legend(loc="lower right")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan04_binary_classification.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualisasi Binary Classification disimpan")
plt.close()

# ============================================================================
# 2. MULTI-CLASS CLASSIFICATION dengan Iris Dataset
# ============================================================================
print("\n2. MULTI-CLASS CLASSIFICATION - Iris Dataset")
print("-" * 60)

# Load Iris dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

print(f"Dataset: {X_iris.shape[0]} samples, {X_iris.shape[1]} features")
print(f"Classes: {iris.target_names}")
print(f"Features: {iris.feature_names}")

# Create DataFrame untuk visualisasi
df_iris = pd.DataFrame(X_iris, columns=iris.feature_names)
df_iris['species'] = pd.Categorical.from_codes(y_iris, iris.target_names)

print("\n5 baris pertama:")
print(df_iris.head())

# Split data
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42
)

# Scaling
scaler_iris = StandardScaler()
X_train_iris_scaled = scaler_iris.fit_transform(X_train_iris)
X_test_iris_scaled = scaler_iris.transform(X_test_iris)

# Train Logistic Regression
log_reg_multi = LogisticRegression(max_iter=200, random_state=42)
log_reg_multi.fit(X_train_iris_scaled, y_train_iris)
y_pred_log = log_reg_multi.predict(X_test_iris_scaled)

# Train Decision Tree
dt_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_clf.fit(X_train_iris, y_train_iris)
y_pred_dt = dt_clf.predict(X_test_iris)

# Evaluasi
print("\n--- Evaluasi Logistic Regression ---")
print(f"Accuracy: {accuracy_score(y_test_iris, y_pred_log):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_iris, y_pred_log, target_names=iris.target_names))

print("\n--- Evaluasi Decision Tree ---")
print(f"Accuracy: {accuracy_score(y_test_iris, y_pred_dt):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_iris, y_pred_dt, target_names=iris.target_names))

# Confusion Matrix
cm_log = confusion_matrix(y_test_iris, y_pred_log)
cm_dt = confusion_matrix(y_test_iris, y_pred_dt)

# Visualisasi
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Multi-class Classification - Iris Dataset', fontsize=16, fontweight='bold')

# Plot 1: Confusion Matrix - Logistic Regression
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=iris.target_names, yticklabels=iris.target_names)
axes[0, 0].set_title('Confusion Matrix - Logistic Regression')
axes[0, 0].set_ylabel('Actual')
axes[0, 0].set_xlabel('Predicted')

# Plot 2: Confusion Matrix - Decision Tree
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens', ax=axes[0, 1],
            xticklabels=iris.target_names, yticklabels=iris.target_names)
axes[0, 1].set_title('Confusion Matrix - Decision Tree')
axes[0, 1].set_ylabel('Actual')
axes[0, 1].set_xlabel('Predicted')

# Plot 3: Feature Importance (Decision Tree)
feature_importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': dt_clf.feature_importances_
}).sort_values('importance', ascending=False)

axes[1, 0].barh(feature_importance['feature'], feature_importance['importance'], color='skyblue')
axes[1, 0].set_xlabel('Importance')
axes[1, 0].set_title('Feature Importance - Decision Tree')
axes[1, 0].invert_yaxis()
axes[1, 0].grid(True, alpha=0.3, axis='x')

# Plot 4: Pairplot (2 features saja untuk simplicity)
for i, species in enumerate(iris.target_names):
    mask = y_iris == i
    axes[1, 1].scatter(X_iris[mask, 0], X_iris[mask, 1], 
                       label=species, alpha=0.6, s=50)
axes[1, 1].set_xlabel(iris.feature_names[0])
axes[1, 1].set_ylabel(iris.feature_names[1])
axes[1, 1].set_title('Feature Visualization (2 features)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan04_multiclass_classification.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualisasi Multi-class Classification disimpan")
plt.close()

# ============================================================================
# 3. DECISION TREE VISUALIZATION
# ============================================================================
print("\n3. DECISION TREE VISUALIZATION")
print("-" * 60)

# Visualisasi Decision Tree
plt.figure(figsize=(16, 10))
plot_tree(dt_clf, 
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Decision Tree Classifier - Iris Dataset', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan04_decision_tree_viz.png', dpi=300, bbox_inches='tight')
print("✓ Visualisasi Decision Tree disimpan")
plt.close()

# ============================================================================
# 4. MODEL COMPARISON
# ============================================================================
print("\n4. MODEL COMPARISON")
print("-" * 60)

# Compare models
models = {
    'Logistic Regression': {
        'accuracy': accuracy_score(y_test_iris, y_pred_log),
        'precision': precision_score(y_test_iris, y_pred_log, average='weighted'),
        'recall': recall_score(y_test_iris, y_pred_log, average='weighted'),
        'f1': f1_score(y_test_iris, y_pred_log, average='weighted')
    },
    'Decision Tree': {
        'accuracy': accuracy_score(y_test_iris, y_pred_dt),
        'precision': precision_score(y_test_iris, y_pred_dt, average='weighted'),
        'recall': recall_score(y_test_iris, y_pred_dt, average='weighted'),
        'f1': f1_score(y_test_iris, y_pred_dt, average='weighted')
    }
}

comparison_df = pd.DataFrame(models).T
print("\nPerbandingan Model:")
print(comparison_df)

# Visualisasi perbandingan
fig, ax = plt.subplots(figsize=(10, 6))
comparison_df.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
ax.set_title('Model Comparison - Classification Metrics', fontsize=14, fontweight='bold')
ax.set_ylabel('Score')
ax.set_xlabel('Model')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.1])

plt.tight_layout()
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan04_model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualisasi Model Comparison disimpan")
plt.close()

# ============================================================================
# 5. PROBABILITY INTERPRETATION
# ============================================================================
print("\n5. PROBABILITY INTERPRETATION")
print("-" * 60)

# Contoh prediksi dengan probabilitas
sample_idx = [0, 1, 2]
samples = X_test_iris_scaled[sample_idx]
actual_labels = y_test_iris[sample_idx]
predictions = log_reg_multi.predict(samples)
probabilities = log_reg_multi.predict_proba(samples)

print("\nContoh Prediksi dengan Probabilitas:")
for i, (actual, pred, proba) in enumerate(zip(actual_labels, predictions, probabilities)):
    print(f"\nSample {i+1}:")
    print(f"  Actual: {iris.target_names[actual]}")
    print(f"  Predicted: {iris.target_names[pred]}")
    print(f"  Probabilities:")
    for j, species in enumerate(iris.target_names):
        print(f"    {species}: {proba[j]:.4f} ({proba[j]*100:.2f}%)")

print("\n" + "=" * 60)
print("SELESAI - Pertemuan 4")
print("=" * 60)
print("\nRingkasan yang dipelajari:")
print("✓ Binary Classification dengan Logistic Regression")
print("✓ Multi-class Classification")
print("✓ Decision Tree Classifier")
print("✓ Confusion Matrix dan metrics (Accuracy, Precision, Recall, F1)")
print("✓ ROC Curve dan AUC")
print("✓ Feature Importance")
print("✓ Model Comparison")
print("\nFile yang dibuat:")
print("- pertemuan04_binary_classification.png")
print("- pertemuan04_multiclass_classification.png")
print("- pertemuan04_decision_tree_viz.png")
print("- pertemuan04_model_comparison.png")
