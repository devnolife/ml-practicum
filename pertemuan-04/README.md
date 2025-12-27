# Pertemuan 4: Classification - Logistic Regression & Decision Tree

## 🎯 Tujuan Pembelajaran

Setelah menyelesaikan pertemuan ini, mahasiswa diharapkan mampu:
1. Memahami perbedaan antara regression dan classification
2. Mengimplementasikan Logistic Regression untuk binary dan multi-class classification
3. Memahami konsep dan implementasi Decision Tree
4. Melakukan evaluasi model classification dengan confusion matrix dan metrics
5. Memahami konsep probabilitas dalam classification
6. Memvisualisasi decision boundary

## 📚 Teori Singkat

### Classification vs Regression

| Aspect | Regression | Classification |
|--------|-----------|----------------|
| Output | Continuous values | Discrete classes/categories |
| Example | Harga rumah, suhu | Spam/Not spam, Species |
| Algorithms | Linear Regression, Polynomial Regression | Logistic Regression, Decision Tree, SVM |

### Logistic Regression

Meskipun namanya "regression", Logistic Regression adalah algoritma **classification**.

**Binary Classification:**
```
p(y=1|x) = 1 / (1 + e^-(β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ))
```

Fungsi sigmoid mengkonversi output linear menjadi probabilitas (0-1).

**Multi-class Classification**: One-vs-Rest (OvR) atau Multinomial

### Decision Tree

Decision Tree adalah model tree-based yang membagi data berdasarkan fitur untuk membuat keputusan.

**Konsep Kunci:**
- **Node**: Titik keputusan
- **Branch**: Hasil keputusan
- **Leaf**: Output (class)
- **Splitting Criteria**: Gini impurity atau Entropy (Information Gain)

**Gini Impurity:**
```
Gini = 1 - Σ(pᵢ)²
```

### Metrics Evaluasi Classification

1. **Accuracy**: (TP + TN) / Total
2. **Precision**: TP / (TP + FP) - "Dari yang diprediksi positif, berapa yang benar?"
3. **Recall (Sensitivity)**: TP / (TP + FN) - "Dari yang sebenarnya positif, berapa yang terdeteksi?"
4. **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
5. **Confusion Matrix**: Tabel TP, TN, FP, FN

## 📝 Praktikum

### Persiapan: Import Library

```python
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
```

### Langkah 1: Binary Classification dengan Logistic Regression

```python
# Generate binary classification dataset
X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    random_state=42
)

# Visualisasi dataset
plt.figure(figsize=(8, 6))
plt.scatter(X[y==0][:, 0], X[y==0][:, 1], c='blue', label='Class 0', alpha=0.6)
plt.scatter(X[y==1][:, 0], X[y==1][:, 1], c='red', label='Class 1', alpha=0.6)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Binary Classification Dataset')
plt.legend()
plt.show()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)

# Predict
y_pred = lr_model.predict(X_test_scaled)
y_pred_proba = lr_model.predict_proba(X_test_scaled)

# Evaluasi
print("=== Logistic Regression - Binary Classification ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Visualisasi Decision Boundary
def plot_decision_boundary(model, X, y, scaler=None):
    h = 0.02  # step size
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
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='RdYlBu')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.figure(figsize=(10, 6))
plot_decision_boundary(lr_model, X_test, y_test, scaler)
plt.title('Decision Boundary - Logistic Regression')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()
```

### Langkah 2: Multi-class Classification - Dataset Iris

```python
# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create DataFrame for better visualization
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y
df['species_name'] = df['species'].map({
    0: 'setosa', 1: 'versicolor', 2: 'virginica'
})

print("=== Dataset Info ===")
print(df.head(10))
print(f"\nClasses: {iris.target_names}")
print(f"Features: {iris.feature_names}")

# Visualisasi dataset (2 features untuk simplicity)
plt.figure(figsize=(10, 6))
colors = ['red', 'green', 'blue']
for i, species in enumerate(iris.target_names):
    plt.scatter(
        df[df['species']==i]['sepal length (cm)'],
        df[df['species']==i]['sepal width (cm)'],
        c=colors[i],
        label=species,
        alpha=0.6
    )
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Iris Dataset - Sepal Features')
plt.legend()
plt.show()

# Gunakan 2 features untuk visualisasi decision boundary
X_2d = X[:, :2]  # Sepal length and width

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_2d, y, test_size=0.2, random_state=42
)

# Train Logistic Regression (multi-class)
lr_multi = LogisticRegression(multi_class='multinomial', max_iter=200, random_state=42)
lr_multi.fit(X_train, y_train)

# Predict
y_pred = lr_multi.predict(X_test)

# Evaluasi
print("\n=== Logistic Regression - Multi-class ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Multi-class Classification')
plt.show()

# Decision Boundary (multi-class)
plt.figure(figsize=(10, 6))
plot_decision_boundary(lr_multi, X_test, y_test)
plt.title('Decision Boundary - Logistic Regression (Multi-class)')
plt.show()
```

### Langkah 3: Decision Tree Classifier

```python
# Train Decision Tree
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)

# Predict
y_pred_dt = dt_model.predict(X_test)

# Evaluasi
print("\n=== Decision Tree Classifier ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred_dt, target_names=iris.target_names))

# Confusion Matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Decision Tree')
plt.show()

# Visualisasi Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dt_model, 
          feature_names=iris.feature_names[:2],
          class_names=iris.target_names,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Decision Tree Visualization')
plt.show()

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': iris.feature_names[:2],
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n=== Feature Importance ===")
print(feature_importance)

plt.figure(figsize=(8, 5))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance - Decision Tree')
plt.tight_layout()
plt.show()

# Decision Boundary
plt.figure(figsize=(10, 6))
plot_decision_boundary(dt_model, X_test, y_test)
plt.title('Decision Boundary - Decision Tree')
plt.show()
```

### Langkah 4: Comparison - Logistic Regression vs Decision Tree

```python
# Train dengan semua features
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression
lr_full = LogisticRegression(multi_class='multinomial', max_iter=200, random_state=42)
lr_full.fit(X_train_full, y_train_full)
y_pred_lr = lr_full.predict(X_test_full)

# Decision Tree
dt_full = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_full.fit(X_train_full, y_train_full)
y_pred_dt = dt_full.predict(X_test_full)

# Comparison
comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree'],
    'Accuracy': [
        accuracy_score(y_test_full, y_pred_lr),
        accuracy_score(y_test_full, y_pred_dt)
    ],
    'Precision': [
        precision_score(y_test_full, y_pred_lr, average='weighted'),
        precision_score(y_test_full, y_pred_dt, average='weighted')
    ],
    'Recall': [
        recall_score(y_test_full, y_pred_lr, average='weighted'),
        recall_score(y_test_full, y_pred_dt, average='weighted')
    ],
    'F1-Score': [
        f1_score(y_test_full, y_pred_lr, average='weighted'),
        f1_score(y_test_full, y_pred_dt, average='weighted')
    ]
})

print("\n=== Model Comparison ===")
print(comparison_df)

# Visualisasi comparison
comparison_df.set_index('Model').plot(kind='bar', figsize=(10, 6), rot=0)
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.ylim(0.7, 1.0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
```

### Langkah 5: Tuning Decision Tree Parameters

```python
# Experiment dengan berbagai max_depth
depths = range(1, 15)
train_scores = []
test_scores = []

for depth in depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train_full, y_train_full)
    
    train_scores.append(accuracy_score(y_train_full, dt.predict(X_train_full)))
    test_scores.append(accuracy_score(y_test_full, dt.predict(X_test_full)))

# Visualisasi
plt.figure(figsize=(10, 6))
plt.plot(depths, train_scores, 'o-', label='Training Accuracy', linewidth=2)
plt.plot(depths, test_scores, 'o-', label='Testing Accuracy', linewidth=2)
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree: Model Complexity vs Performance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nOptimal max_depth: {depths[np.argmax(test_scores)]}")
print(f"Best test accuracy: {max(test_scores):.4f}")
```

## 💪 Tugas Praktikum

### Tugas 1: Binary Classification - Breast Cancer (30 poin)

Gunakan dataset Breast Cancer dari scikit-learn:

```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
```

Lakukan:
1. Load dan eksplorasi dataset
2. Split data (80-20)
3. Train Logistic Regression model
4. Evaluasi dengan confusion matrix, accuracy, precision, recall, F1-score
5. Plot ROC curve dan hitung AUC
6. Visualisasi decision boundary (gunakan 2 features terbaik)
7. Interpretasi hasil: Apakah model lebih baik dalam precision atau recall? Mengapa ini penting untuk kasus breast cancer?

### Tugas 2: Multi-class Classification - Wine Dataset (35 poin)

Gunakan dataset Wine dari scikit-learn:

```python
from sklearn.datasets import load_wine
wine = load_wine()
```

Lakukan:
1. EDA singkat (distribusi kelas, correlation)
2. Train Logistic Regression (multi-class)
3. Train Decision Tree
4. Bandingkan performance kedua model
5. Untuk Decision Tree:
   - Visualisasi tree
   - Analisis feature importance
   - Experiment dengan max_depth (1-10)
6. Buat kesimpulan: Model mana yang lebih baik untuk dataset ini?

### Tugas 3: Overfitting Analysis (20 poin)

Gunakan dataset yang sama dari Tugas 2:
1. Train Decision Tree dengan max_depth dari 1 sampai 20
2. Plot learning curve (train vs test accuracy)
3. Identifikasi titik optimal dan kapan terjadi overfitting
4. Bandingkan dengan Logistic Regression (tidak ada parameter depth)
5. Jelaskan mengapa Decision Tree lebih prone ke overfitting

### Tugas 4: Custom Dataset Classification (15 poin)

Buat atau download dataset untuk salah satu kasus:
- Email spam classification
- Customer churn prediction  
- Titanic survival prediction

Lakukan:
1. Preprocessing (handling missing values, encoding categorical variables)
2. Train minimal 2 model (Logistic Regression dan Decision Tree)
3. Evaluasi dan bandingkan
4. Interpretasi dan rekomendasi untuk improvement

## 📤 Cara Mengumpulkan

1. Notebook lengkap dengan semua tugas
2. Setiap section ada interpretasi hasil
3. Export ke PDF: `NIM_Nama_Pertemuan04.pdf`
4. Upload ke LMS atau GitHub

## ✅ Kriteria Penilaian

| Aspek | Bobot |
|-------|-------|
| Tugas 1: Binary Classification | 30% |
| Tugas 2: Multi-class Classification | 35% |
| Tugas 3: Overfitting Analysis | 20% |
| Tugas 4: Custom Dataset | 15% |
| Interpretasi confusion matrix & metrics | 20% |
| Visualisasi decision boundary/tree | 15% |
| Dokumentasi | 10% |

## 📚 Referensi

1. [Logistic Regression - Scikit-learn](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
2. [Decision Trees - Scikit-learn](https://scikit-learn.org/stable/modules/tree.html)
3. [Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
4. [Understanding ROC Curves](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)

## 💡 Tips

- **Imbalanced data?** Perhatikan precision dan recall, jangan hanya accuracy!
- **Decision Tree overfitting?** Gunakan max_depth, min_samples_split, min_samples_leaf
- **Logistic Regression tidak converge?** Increase max_iter atau standardize features
- **Multi-class?** Gunakan average='weighted' untuk metrics

---

**Happy Classifying! 🎯🌸**
