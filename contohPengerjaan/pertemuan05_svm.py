"""
Pertemuan 5: Support Vector Machine (SVM)
Contoh program lengkap untuk Linear SVM, Non-linear SVM, dan Parameter Tuning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, make_circles, make_moons, load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Settings
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 60)
print("PERTEMUAN 5: Support Vector Machine (SVM)")
print("=" * 60)

# ============================================================================
# 1. LINEAR SVM
# ============================================================================
print("\n1. LINEAR SVM")
print("-" * 60)

# Generate linearly separable data
np.random.seed(42)
X_linear, y_linear = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=2.0,
    random_state=42
)

print(f"Dataset: {X_linear.shape[0]} samples, {X_linear.shape[1]} features")

# Split data
X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(
    X_linear, y_linear, test_size=0.3, random_state=42
)

# PENTING: Scaling untuk SVM!
scaler_lin = StandardScaler()
X_train_lin_scaled = scaler_lin.fit_transform(X_train_lin)
X_test_lin_scaled = scaler_lin.transform(X_test_lin)

# Train Linear SVM
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X_train_lin_scaled, y_train_lin)

# Predict
y_pred_lin = svm_linear.predict(X_test_lin_scaled)

# Evaluasi
accuracy_lin = accuracy_score(y_test_lin, y_pred_lin)
print(f"\nAccuracy: {accuracy_lin:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_lin, y_pred_lin))

# Support Vectors
print(f"\nJumlah Support Vectors: {svm_linear.n_support_}")
print(f"  Class 0: {svm_linear.n_support_[0]}")
print(f"  Class 1: {svm_linear.n_support_[1]}")

# Visualisasi Decision Boundary
def plot_decision_boundary(X, y, model, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, 
                        edgecolors='black', s=50)
    
    # Plot support vectors
    if hasattr(model, 'support_vectors_'):
        ax.scatter(model.support_vectors_[:, 0], 
                  model.support_vectors_[:, 1],
                  s=200, linewidth=1.5, facecolors='none', 
                  edgecolors='green', label='Support Vectors')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.figure(figsize=(10, 6))
plot_decision_boundary(X_train_lin_scaled, y_train_lin, svm_linear, 
                       f'Linear SVM (Accuracy: {accuracy_lin:.4f})')
plt.tight_layout()
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan05_linear_svm.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualisasi Linear SVM disimpan")
plt.close()

# ============================================================================
# 2. NON-LINEAR SVM dengan RBF Kernel
# ============================================================================
print("\n2. NON-LINEAR SVM - RBF Kernel")
print("-" * 60)

# Generate non-linearly separable data (circles)
X_circles, y_circles = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)

print(f"Dataset: {X_circles.shape[0]} samples (circular pattern)")

# Split data
X_train_circ, X_test_circ, y_train_circ, y_test_circ = train_test_split(
    X_circles, y_circles, test_size=0.3, random_state=42
)

# Scaling
scaler_circ = StandardScaler()
X_train_circ_scaled = scaler_circ.fit_transform(X_train_circ)
X_test_circ_scaled = scaler_circ.transform(X_test_circ)

# Train Linear SVM (untuk perbandingan)
svm_linear_circ = SVC(kernel='linear', random_state=42)
svm_linear_circ.fit(X_train_circ_scaled, y_train_circ)
y_pred_linear_circ = svm_linear_circ.predict(X_test_circ_scaled)
accuracy_linear_circ = accuracy_score(y_test_circ, y_pred_linear_circ)

# Train RBF SVM
svm_rbf = SVC(kernel='rbf', gamma='auto', C=1.0, random_state=42)
svm_rbf.fit(X_train_circ_scaled, y_train_circ)
y_pred_rbf = svm_rbf.predict(X_test_circ_scaled)
accuracy_rbf = accuracy_score(y_test_circ, y_pred_rbf)

print(f"\nLinear SVM Accuracy: {accuracy_linear_circ:.4f}")
print(f"RBF SVM Accuracy: {accuracy_rbf:.4f}")
print(f"Improvement: {(accuracy_rbf - accuracy_linear_circ)*100:.2f}%")

# Visualisasi perbandingan
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Linear vs RBF Kernel - Circular Data', fontsize=16, fontweight='bold')

plot_decision_boundary(X_train_circ_scaled, y_train_circ, svm_linear_circ, 
                       f'Linear Kernel (Acc: {accuracy_linear_circ:.4f})', ax=axes[0])
plot_decision_boundary(X_train_circ_scaled, y_train_circ, svm_rbf, 
                       f'RBF Kernel (Acc: {accuracy_rbf:.4f})', ax=axes[1])

plt.tight_layout()
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan05_rbf_kernel.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualisasi RBF Kernel disimpan")
plt.close()

# ============================================================================
# 3. DIFFERENT KERNELS COMPARISON
# ============================================================================
print("\n3. PERBANDINGAN BERBAGAI KERNEL")
print("-" * 60)

# Generate moons dataset
X_moons, y_moons = make_moons(n_samples=300, noise=0.15, random_state=42)

# Split dan scale
X_train_moon, X_test_moon, y_train_moon, y_test_moon = train_test_split(
    X_moons, y_moons, test_size=0.3, random_state=42
)

scaler_moon = StandardScaler()
X_train_moon_scaled = scaler_moon.fit_transform(X_train_moon)
X_test_moon_scaled = scaler_moon.transform(X_test_moon)

# Train dengan berbagai kernel
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
models = {}
accuracies = {}

for kernel in kernels:
    print(f"\nTraining SVM with {kernel} kernel...")
    model = SVC(kernel=kernel, random_state=42)
    model.fit(X_train_moon_scaled, y_train_moon)
    y_pred = model.predict(X_test_moon_scaled)
    accuracy = accuracy_score(y_test_moon, y_pred)
    
    models[kernel] = model
    accuracies[kernel] = accuracy
    print(f"  Accuracy: {accuracy:.4f}")

# Visualisasi semua kernel
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('SVM Kernel Comparison - Moons Dataset', fontsize=16, fontweight='bold')

for idx, kernel in enumerate(kernels):
    row = idx // 2
    col = idx % 2
    plot_decision_boundary(X_train_moon_scaled, y_train_moon, models[kernel], 
                          f'{kernel.capitalize()} Kernel (Acc: {accuracies[kernel]:.4f})',
                          ax=axes[row, col])

plt.tight_layout()
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan05_kernel_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualisasi Kernel Comparison disimpan")
plt.close()

# ============================================================================
# 4. HYPERPARAMETER TUNING
# ============================================================================
print("\n4. HYPERPARAMETER TUNING dengan GridSearchCV")
print("-" * 60)

# Load Breast Cancer dataset
cancer = load_breast_cancer()
X_cancer = cancer.data
y_cancer = cancer.target

print(f"Breast Cancer Dataset: {X_cancer.shape[0]} samples, {X_cancer.shape[1]} features")
print(f"Classes: {cancer.target_names}")

# Split
X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = train_test_split(
    X_cancer, y_cancer, test_size=0.3, random_state=42
)

# Scaling
scaler_cancer = StandardScaler()
X_train_cancer_scaled = scaler_cancer.fit_transform(X_train_cancer)
X_test_cancer_scaled = scaler_cancer.transform(X_test_cancer)

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf']
}

print("\nParameter Grid:")
print(f"  C: {param_grid['C']}")
print(f"  gamma: {param_grid['gamma']}")
print(f"  kernel: {param_grid['kernel']}")

print("\nMelakukan Grid Search (ini bisa memakan waktu)...")

# GridSearchCV
grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, 
                          scoring='accuracy', n_jobs=-1, verbose=0)
grid_search.fit(X_train_cancer_scaled, y_train_cancer)

print("\n✓ Grid Search selesai!")

# Best parameters
print("\nBest Parameters:")
print(f"  C: {grid_search.best_params_['C']}")
print(f"  gamma: {grid_search.best_params_['gamma']}")
print(f"  kernel: {grid_search.best_params_['kernel']}")
print(f"\nBest CV Score: {grid_search.best_score_:.4f}")

# Evaluate on test set
best_model = grid_search.best_estimator_
y_pred_cancer = best_model.predict(X_test_cancer_scaled)
test_accuracy = accuracy_score(y_test_cancer, y_pred_cancer)

print(f"Test Set Accuracy: {test_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test_cancer, y_pred_cancer, target_names=cancer.target_names))

# Confusion Matrix
cm_cancer = confusion_matrix(y_test_cancer, y_pred_cancer)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('SVM on Breast Cancer Dataset', fontsize=16, fontweight='bold')

# Plot 1: Confusion Matrix
sns.heatmap(cm_cancer, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=cancer.target_names, yticklabels=cancer.target_names)
axes[0].set_title(f'Confusion Matrix (Acc: {test_accuracy:.4f})')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# Plot 2: GridSearch Results (C vs Gamma heatmap)
# Create results dataframe
results = pd.DataFrame(grid_search.cv_results_)
results_pivot = results.pivot_table(
    values='mean_test_score',
    index='param_gamma',
    columns='param_C'
)

sns.heatmap(results_pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1])
axes[1].set_title('Grid Search Results (CV Accuracy)')
axes[1].set_xlabel('C')
axes[1].set_ylabel('gamma')

plt.tight_layout()
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan05_hyperparameter_tuning.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualisasi Hyperparameter Tuning disimpan")
plt.close()

# ============================================================================
# 5. EFFECT OF C PARAMETER
# ============================================================================
print("\n5. PENGARUH PARAMETER C (Regularization)")
print("-" * 60)

# Generate data
X_reg, y_reg = make_classification(n_samples=100, n_features=2, n_redundant=0,
                                   n_clusters_per_class=1, random_state=42)

# Scale
scaler_reg = StandardScaler()
X_reg_scaled = scaler_reg.fit_transform(X_reg)

# Different C values
C_values = [0.1, 1, 10, 100]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Effect of C Parameter (Regularization)', fontsize=16, fontweight='bold')

for idx, C in enumerate(C_values):
    row = idx // 2
    col = idx % 2
    
    model = SVC(kernel='linear', C=C, random_state=42)
    model.fit(X_reg_scaled, y_reg)
    
    plot_decision_boundary(X_reg_scaled, y_reg, model,
                          f'C = {C} (Support Vectors: {model.n_support_.sum()})',
                          ax=axes[row, col])

plt.tight_layout()
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan05_c_parameter_effect.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualisasi C Parameter Effect disimpan")
plt.close()

print("\nInterpretasi Parameter C:")
print("  C kecil (0.1): Margin lebih besar, lebih banyak support vectors")
print("  C besar (100): Margin lebih kecil, lebih sedikit support vectors, risk overfitting")

print("\n" + "=" * 60)
print("SELESAI - Pertemuan 5")
print("=" * 60)
print("\nRingkasan yang dipelajari:")
print("✓ Linear SVM untuk data linearly separable")
print("✓ Non-linear SVM dengan RBF kernel")
print("✓ Perbandingan berbagai kernel (linear, poly, rbf, sigmoid)")
print("✓ Hyperparameter tuning dengan GridSearchCV")
print("✓ Pengaruh parameter C dan gamma")
print("✓ Support vectors visualization")
print("\nFile yang dibuat:")
print("- pertemuan05_linear_svm.png")
print("- pertemuan05_rbf_kernel.png")
print("- pertemuan05_kernel_comparison.png")
print("- pertemuan05_hyperparameter_tuning.png")
print("- pertemuan05_c_parameter_effect.png")
