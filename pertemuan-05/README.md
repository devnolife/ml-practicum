# Pertemuan 5: Support Vector Machine (SVM)

## 🎯 Tujuan Pembelajaran

Setelah menyelesaikan pertemuan ini, mahasiswa diharapkan mampu:
1. Memahami konsep dasar Support Vector Machine (SVM)
2. Memahami perbedaan linear dan non-linear SVM
3. Mengimplementasikan SVM dengan berbagai kernel
4. Memahami konsep margin, support vectors, dan hyperplane
5. Melakukan parameter tuning untuk SVM (C dan gamma)
6. Membandingkan performa SVM dengan algoritma lain

## 📚 Teori Singkat

### Apa itu Support Vector Machine?

SVM adalah algoritma supervised learning yang mencari **hyperplane terbaik** untuk memisahkan data dari berbagai kelas.

**Konsep Kunci:**
- **Hyperplane**: Bidang pemisah antar kelas
- **Margin**: Jarak antara hyperplane dan data point terdekat
- **Support Vectors**: Data points yang berada di margin (paling dekat dengan hyperplane)
- **Tujuan SVM**: Maksimalkan margin

### Linear vs Non-Linear SVM

**Linear SVM:**
```
Data dapat dipisahkan dengan garis lurus (2D) atau bidang (3D+)
```

**Non-Linear SVM:**
```
Menggunakan kernel trick untuk transform data ke dimensi lebih tinggi
Contoh: data yang membentuk lingkaran (XOR problem)
```

### Kernel Functions

1. **Linear Kernel**: `K(x, y) = x^T · y`
   - Untuk data yang linearly separable
   - Paling cepat, paling simple

2. **Polynomial Kernel**: `K(x, y) = (γx^T·y + r)^d`
   - Untuk data dengan hubungan polynomial
   - Parameter: degree (d), gamma (γ), coef0 (r)

3. **RBF (Radial Basis Function) Kernel**: `K(x, y) = exp(-γ||x-y||²)`
   - Paling popular untuk non-linear
   - Parameter: gamma (γ)
   - Default choice untuk banyak problem

4. **Sigmoid Kernel**: `K(x, y) = tanh(γx^T·y + r)`
   - Mirip neural network
   - Jarang digunakan

### Parameter SVM

1. **C (Regularization Parameter)**:
   - C kecil: Margin lebih besar, toleran terhadap misclassification
   - C besar: Margin lebih kecil, strict classification

2. **Gamma (untuk RBF kernel)**:
   - Gamma kecil: Decision boundary smooth
   - Gamma besar: Decision boundary lebih complex, risk overfitting

## 📝 Praktikum

### Persiapan: Import Library

```python
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
```

### Langkah 1: Linear SVM

```python
# Generate linearly separable data
X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    class_sep=2,
    random_state=42
)

# Visualisasi data
plt.figure(figsize=(8, 6))
plt.scatter(X[y==0][:, 0], X[y==0][:, 1], c='blue', label='Class 0', alpha=0.6, edgecolors='k')
plt.scatter(X[y==1][:, 0], X[y==1][:, 1], c='red', label='Class 1', alpha=0.6, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Linearly Separable Dataset')
plt.legend()
plt.show()

# Split dan scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Linear SVM
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X_train_scaled, y_train)

# Predict
y_pred = svm_linear.predict(X_test_scaled)

# Evaluasi
print("=== Linear SVM ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\nNumber of support vectors: {len(svm_linear.support_vectors_)}")
print(f"Support vector indices: {svm_linear.support_}")

# Visualisasi Decision Boundary dengan Support Vectors
def plot_svm_boundary(model, X, y, scaler=None, title='SVM Decision Boundary'):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    mesh_input = np.c_[xx.ravel(), yy.ravel()]
    if scaler:
        mesh_input = scaler.transform(mesh_input)
    
    Z = model.predict(mesh_input)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='RdYlBu', s=50)
    
    # Plot support vectors
    if scaler:
        sv = scaler.inverse_transform(model.support_vectors_)
    else:
        sv = model.support_vectors_
    plt.scatter(sv[:, 0], sv[:, 1], s=200, linewidth=1.5, 
                facecolors='none', edgecolors='k', label='Support Vectors')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()

plt.figure(figsize=(10, 8))
plot_svm_boundary(svm_linear, X_test, y_test, scaler, 'Linear SVM - Decision Boundary')
plt.show()
```

### Langkah 2: Non-Linear SVM - RBF Kernel

```python
# Generate non-linear data (circles)
X_circles, y_circles = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)

# Visualisasi
plt.figure(figsize=(8, 6))
plt.scatter(X_circles[y_circles==0][:, 0], X_circles[y_circles==0][:, 1], 
           c='blue', label='Class 0', alpha=0.6, edgecolors='k')
plt.scatter(X_circles[y_circles==1][:, 0], X_circles[y_circles==1][:, 1], 
           c='red', label='Class 1', alpha=0.6, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Non-Linear Dataset (Circles)')
plt.legend()
plt.axis('equal')
plt.show()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_circles, y_circles, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare Linear vs RBF kernel
kernels = ['linear', 'rbf']
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, kernel in enumerate(kernels):
    # Train SVM
    svm = SVC(kernel=kernel, C=1.0, gamma='scale', random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Plot
    plt.sca(axes[idx])
    plot_svm_boundary(svm, X_test, y_test, scaler, 
                     f'{kernel.upper()} Kernel (Accuracy: {accuracy:.3f})')
    plt.axis('equal')

plt.tight_layout()
plt.show()

print("=== Comparison ===")
print(f"Linear kernel: Less flexible, cannot separate circular data well")
print(f"RBF kernel: More flexible, can handle non-linear boundaries")
```

### Langkah 3: Effect of C Parameter

```python
# Generate data
X_moons, y_moons = make_moons(n_samples=200, noise=0.15, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X_moons, y_moons, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Test different C values
C_values = [0.1, 1, 10, 100]
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

for idx, C in enumerate(C_values):
    svm = SVC(kernel='rbf', C=C, gamma='scale', random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    y_pred = svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    plt.sca(axes[idx//2, idx%2])
    plot_svm_boundary(svm, X_test, y_test, scaler, 
                     f'C = {C} (Accuracy: {accuracy:.3f}, SV: {len(svm.support_vectors_)})')

plt.tight_layout()
plt.show()

print("\n=== Effect of C Parameter ===")
print("C = 0.1:  Large margin, more misclassification, many support vectors")
print("C = 1:    Balanced")
print("C = 10:   Smaller margin, fewer misclassification")
print("C = 100:  Very small margin, risk of overfitting, fewer support vectors")
```

### Langkah 4: Effect of Gamma Parameter (RBF Kernel)

```python
# Test different gamma values
gamma_values = [0.01, 0.1, 1, 10]
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

for idx, gamma in enumerate(gamma_values):
    svm = SVC(kernel='rbf', C=1.0, gamma=gamma, random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    y_pred = svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    plt.sca(axes[idx//2, idx%2])
    plot_svm_boundary(svm, X_test, y_test, scaler,
                     f'Gamma = {gamma} (Accuracy: {accuracy:.3f})')

plt.tight_layout()
plt.show()

print("\n=== Effect of Gamma Parameter ===")
print("Gamma = 0.01: Smooth decision boundary, may underfit")
print("Gamma = 0.1:  Balanced")
print("Gamma = 1:    More complex boundary")
print("Gamma = 10:   Very complex boundary, risk of overfitting")
```

### Langkah 5: Comparison of Different Kernels

```python
# Load dataset
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Test different kernels
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

results = []

for idx, kernel in enumerate(kernels):
    svm = SVC(kernel=kernel, C=1.0, gamma='scale', random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    y_pred = svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    results.append({
        'Kernel': kernel,
        'Accuracy': accuracy,
        'Support Vectors': len(svm.support_vectors_)
    })
    
    plt.sca(axes[idx//2, idx%2])
    plot_svm_boundary(svm, X_test, y_test, scaler,
                     f'{kernel.upper()} Kernel (Acc: {accuracy:.3f})')

plt.tight_layout()
plt.show()

# Summary table
results_df = pd.DataFrame(results)
print("\n=== Kernel Comparison ===")
print(results_df.to_string(index=False))
```

### Langkah 6: Hyperparameter Tuning dengan Grid Search

```python
# Load Breast Cancer dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}

# Grid Search
print("Performing Grid Search...")
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Best parameters
print(f"\n=== Grid Search Results ===")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-validation Score: {grid_search.best_score_:.4f}")

# Test dengan best model
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test_scaled)

print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=cancer.target_names,
            yticklabels=cancer.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Optimized SVM')
plt.show()

# Visualisasi Grid Search results
results = pd.DataFrame(grid_search.cv_results_)
pivot_table = results.pivot_table(values='mean_test_score', 
                                   index='param_gamma', 
                                   columns='param_C')

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd')
plt.title('Grid Search Results: Accuracy Scores')
plt.xlabel('C')
plt.ylabel('Gamma')
plt.show()
```

## 💪 Tugas Praktikum

### Tugas 1: SVM untuk Binary Classification (25 poin)

Gunakan dataset Breast Cancer:
1. Load dan split data (80-20)
2. Standardize features
3. Train SVM dengan kernel linear, poly, dan rbf
4. Bandingkan accuracy, precision, recall ketiga kernel
5. Visualisasi confusion matrix
6. Pilih kernel terbaik dan jelaskan alasannya

### Tugas 2: Parameter Tuning Analysis (30 poin)

Gunakan dataset yang sama atau dataset Wine:
1. Experiment dengan C values: [0.01, 0.1, 1, 10, 100]
2. Experiment dengan gamma values: [0.001, 0.01, 0.1, 1, 10]
3. Untuk setiap kombinasi, catat:
   - Training accuracy
   - Testing accuracy
   - Number of support vectors
4. Buat heatmap untuk visualisasi hasil
5. Identifikasi kombinasi terbaik
6. Analisis: Kapan terjadi overfitting? Kapan underfitting?

### Tugas 3: Kernel Comparison (25 poin)

Buat atau gunakan dataset non-linear (circles, moons, atau XOR):
1. Generate 3 different non-linear datasets
2. Untuk setiap dataset, test dengan 4 kernels (linear, poly, rbf, sigmoid)
3. Visualisasi decision boundary untuk semua kombinasi (3x4 = 12 plots)
4. Analisis: Kernel mana yang paling baik untuk setiap jenis dataset?
5. Kesimpulan tentang karakteristik setiap kernel

### Tugas 4: Grid Search Optimization (20 poin)

Pilih dataset classification (bebas, bisa dari Kaggle):
1. Preprocessing yang diperlukan
2. Setup parameter grid untuk C, gamma, dan kernel
3. Lakukan Grid Search dengan cross-validation
4. Visualisasi hasil Grid Search (heatmap atau line plot)
5. Train model dengan best parameters
6. Evaluasi pada test set
7. Interpretasi: Apakah hasil Grid Search memberikan improvement signifikan?

## 📤 Cara Mengumpulkan

1. Notebook dengan semua tugas dan analisis
2. Setiap visualisasi harus ada interpretasi
3. Export ke PDF: `NIM_Nama_Pertemuan05.pdf`
4. Upload ke LMS atau GitHub

## ✅ Kriteria Penilaian

| Aspek | Bobot |
|-------|-------|
| Tugas 1: Kernel Comparison | 25% |
| Tugas 2: Parameter Tuning Analysis | 30% |
| Tugas 3: Non-linear Dataset | 25% |
| Tugas 4: Grid Search | 20% |
| Visualisasi decision boundary | 20% |
| Interpretasi parameter effects | 15% |
| Dokumentasi | 10% |

## 📚 Referensi

1. [SVM - Scikit-learn](https://scikit-learn.org/stable/modules/svm.html)
2. [Understanding the Kernel Trick](https://towardsdatascience.com/understanding-the-kernel-trick-e0bc6112ef78)
3. [SVM Parameters Explained](https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769)
4. [Grid Search CV](https://scikit-learn.org/stable/modules/grid_search.html)

## 💡 Tips

- **Selalu standardize features** untuk SVM!
- **RBF kernel** adalah default choice yang bagus untuk kebanyakan problem
- **C kecil** = soft margin (tolerant), **C besar** = hard margin (strict)
- **Gamma kecil** = simple boundary, **Gamma besar** = complex boundary
- **Grid Search** bisa lama, gunakan `n_jobs=-1` untuk parallel processing

---

**Happy Support Vector Learning! 🎯📐**
