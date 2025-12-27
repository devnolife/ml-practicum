# Pertemuan 6: Ensemble Methods - Random Forest & Gradient Boosting

## 🎯 Tujuan Pembelajaran

Setelah menyelesaikan pertemuan ini, mahasiswa diharapkan mampu:
1. Memahami konsep ensemble learning dan jenisnya (bagging, boosting)
2. Mengimplementasikan Random Forest untuk classification dan regression
3. Memahami dan mengimplementasikan Gradient Boosting
4. Melakukan feature importance analysis
5. Melakukan hyperparameter tuning untuk ensemble models
6. Membandingkan performa single model vs ensemble models

## 📚 Teori Singkat

### Ensemble Learning

**Filosofi**: "The wisdom of crowds" - Kombinasi dari banyak model weak learners menghasilkan strong learner.

```
Ensemble = Kombinasi dari multiple models
Tujuan: Reduce variance, bias, dan meningkatkan prediksi
```

### Jenis Ensemble Methods

**1. Bagging (Bootstrap Aggregating)**
- Train multiple models secara **paralel** dengan data yang berbeda (bootstrap sampling)
- Agregasi hasil dengan voting (classification) atau averaging (regression)
- Contoh: **Random Forest**
- Reduce: **Variance** (overfitting)

**2. Boosting**
- Train multiple models secara **sequential**
- Model berikutnya fokus pada error model sebelumnya
- Contoh: **AdaBoost, Gradient Boosting, XGBoost**
- Reduce: **Bias** (underfitting)

### Random Forest

Random Forest = Ensemble dari banyak Decision Trees dengan:
- **Bootstrap sampling** untuk data
- **Random feature selection** untuk splitting
- **Voting/Averaging** untuk final prediction

**Keuntungan:**
- Sangat accurate
- Handle missing values dan outliers
- Tidak mudah overfit
- Built-in feature importance

**Parameters:**
- `n_estimators`: Jumlah trees
- `max_depth`: Kedalaman maksimal tree
- `max_features`: Jumlah features untuk splitting
- `min_samples_split`: Minimal samples untuk split node

### Gradient Boosting

Boosting yang menggunakan gradient descent untuk meminimalkan loss function.

**Cara Kerja:**
1. Start dengan model sederhana (weak learner)
2. Hitung residual (error)
3. Train model baru untuk predict residual
4. Update prediction = old prediction + learning_rate × new prediction
5. Repeat

**Parameters:**
- `n_estimators`: Jumlah boosting stages
- `learning_rate`: Shrinkage factor
- `max_depth`: Tree depth
- `subsample`: Fraction of samples untuk training

## 📝 Praktikum

### Persiapan: Import Library

```python
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
from sklearn.datasets import load_breast_cancer, load_diabetes, make_classification

# Settings
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
```

### Langkah 1: Random Forest untuk Classification

```python
# Load dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train single Decision Tree (baseline)
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

print("=== Model Comparison ===")
print(f"Decision Tree Accuracy: {acc_dt:.4f}")
print(f"Random Forest Accuracy: {acc_rf:.4f}")
print(f"Improvement: {(acc_rf - acc_dt)*100:.2f}%")

# Classification Report
print("\n=== Random Forest - Classification Report ===")
print(classification_report(y_test, y_pred_rf, target_names=cancer.target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=cancer.target_names,
            yticklabels=cancer.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest')
plt.show()
```

### Langkah 2: Feature Importance Analysis

```python
# Get feature importances
feature_importance = pd.DataFrame({
    'Feature': cancer.feature_names,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n=== Top 10 Most Important Features ===")
print(feature_importance.head(10))

# Visualisasi
plt.figure(figsize=(12, 8))
plt.barh(feature_importance['Feature'][:15], feature_importance['Importance'][:15])
plt.xlabel('Importance')
plt.title('Feature Importance - Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Train model dengan top features
top_features = feature_importance['Feature'][:10].values
feature_indices = [list(cancer.feature_names).index(f) for f in top_features]

X_train_top = X_train[:, feature_indices]
X_test_top = X_test[:, feature_indices]

rf_top = RandomForestClassifier(n_estimators=100, random_state=42)
rf_top.fit(X_train_top, y_train)
y_pred_top = rf_top.predict(X_test_top)
acc_top = accuracy_score(y_test, y_pred_top)

print(f"\n=== Performance Comparison ===")
print(f"All features (30): {acc_rf:.4f}")
print(f"Top 10 features: {acc_top:.4f}")
print(f"Difference: {abs(acc_rf - acc_top):.4f}")
```

### Langkah 3: Effect of n_estimators

```python
# Test berbagai jumlah trees
n_estimators_range = range(10, 201, 10)
train_scores = []
test_scores = []

for n_est in n_estimators_range:
    rf_temp = RandomForestClassifier(n_estimators=n_est, random_state=42, n_jobs=-1)
    rf_temp.fit(X_train, y_train)
    
    train_scores.append(accuracy_score(y_train, rf_temp.predict(X_train)))
    test_scores.append(accuracy_score(y_test, rf_temp.predict(X_test)))

# Visualisasi
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, train_scores, 'o-', label='Training Accuracy', linewidth=2)
plt.plot(n_estimators_range, test_scores, 'o-', label='Testing Accuracy', linewidth=2)
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('Accuracy')
plt.title('Random Forest: Effect of Number of Trees')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

optimal_n = n_estimators_range[np.argmax(test_scores)]
print(f"Optimal number of trees: {optimal_n}")
print(f"Best test accuracy: {max(test_scores):.4f}")
```

### Langkah 4: Gradient Boosting untuk Classification

```python
# Train Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
acc_gb = accuracy_score(y_test, y_pred_gb)

print("=== Gradient Boosting ===")
print(f"Accuracy: {acc_gb:.4f}")
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred_gb, target_names=cancer.target_names))

# Feature Importance
feature_importance_gb = pd.DataFrame({
    'Feature': cancer.feature_names,
    'Importance': gb.feature_importances_
}).sort_values('Importance', ascending=False)

# Compare feature importance: RF vs GB
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

axes[0].barh(feature_importance['Feature'][:15], feature_importance['Importance'][:15])
axes[0].set_xlabel('Importance')
axes[0].set_title('Feature Importance - Random Forest')
axes[0].invert_yaxis()

axes[1].barh(feature_importance_gb['Feature'][:15], feature_importance_gb['Importance'][:15], color='orange')
axes[1].set_xlabel('Importance')
axes[1].set_title('Feature Importance - Gradient Boosting')
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()
```

### Langkah 5: Effect of Learning Rate (Gradient Boosting)

```python
# Test berbagai learning rates
learning_rates = [0.01, 0.05, 0.1, 0.2, 0.5]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for idx, lr in enumerate(learning_rates):
    # Train with different learning rates
    gb_temp = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=lr,
        max_depth=3,
        random_state=42
    )
    gb_temp.fit(X_train, y_train)
    
    # Track training progress
    train_scores_gb = []
    test_scores_gb = []
    
    for i, y_pred in enumerate(gb_temp.staged_predict(X_train)):
        train_scores_gb.append(accuracy_score(y_train, y_pred))
    
    for i, y_pred in enumerate(gb_temp.staged_predict(X_test)):
        test_scores_gb.append(accuracy_score(y_test, y_pred))
    
    # Plot
    axes[idx].plot(train_scores_gb, label='Train', linewidth=2)
    axes[idx].plot(test_scores_gb, label='Test', linewidth=2)
    axes[idx].set_xlabel('Boosting Iterations')
    axes[idx].set_ylabel('Accuracy')
    axes[idx].set_title(f'Learning Rate = {lr}')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

# Remove extra subplot
fig.delaxes(axes[5])
plt.tight_layout()
plt.show()

print("=== Learning Rate Analysis ===")
print("LR = 0.01: Slow learning, may need more iterations")
print("LR = 0.1:  Good balance")
print("LR = 0.5:  Fast learning, risk of overfitting")
```

### Langkah 6: Random Forest untuk Regression

```python
# Load diabetes dataset (regression)
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

# Predict
y_pred_train = rf_reg.predict(X_train)
y_pred_test = rf_reg.predict(X_test)

# Evaluasi
print("=== Random Forest Regressor ===")
print(f"Train R² Score: {r2_score(y_train, y_pred_train):.4f}")
print(f"Test R² Score: {r2_score(y_test, y_pred_test):.4f}")
print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.2f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f}")

# Visualisasi Predicted vs Actual
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Training Set - Random Forest Regression')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, alpha=0.5, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Testing Set - Random Forest Regression')

plt.tight_layout()
plt.show()

# Feature Importance
feature_importance_reg = pd.DataFrame({
    'Feature': diabetes.feature_names,
    'Importance': rf_reg.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_reg['Feature'], feature_importance_reg['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance - Diabetes Prediction')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

### Langkah 7: Hyperparameter Tuning dengan Grid Search

```python
# Parameter grid untuk Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print("Performing Grid Search for Random Forest...")
grid_search_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Use smaller dataset for faster training
X_train_small = X_train[:1000]
y_train_small = y_train[:1000]

grid_search_rf.fit(X_train_small, y_train_small)

print(f"\n=== Random Forest - Best Parameters ===")
print(grid_search_rf.best_params_)
print(f"Best CV Score: {grid_search_rf.best_score_:.4f}")

# Test best model
best_rf = grid_search_rf.best_estimator_
best_rf.fit(X_train, y_train)  # Train on full training set
y_pred_best = best_rf.predict(X_test)

print(f"Test Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")

# Parameter grid untuk Gradient Boosting
param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

print("\nPerforming Grid Search for Gradient Boosting...")
grid_search_gb = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid_gb,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search_gb.fit(X_train_small, y_train_small)

print(f"\n=== Gradient Boosting - Best Parameters ===")
print(grid_search_gb.best_params_)
print(f"Best CV Score: {grid_search_gb.best_score_:.4f}")
```

### Langkah 8: Model Comparison - Customer Churn Prediction

```python
# Generate synthetic churn dataset
X_churn, y_churn = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    weights=[0.7, 0.3],  # Imbalanced
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X_churn, y_churn, test_size=0.2, random_state=42
)

# Train multiple models
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = []

for name, model in models.items():
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    # Train and test
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results.append({
        'Model': name,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std(),
        'Test Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results)
print("\n=== Model Comparison - Customer Churn ===")
print(results_df.to_string(index=False))

# Visualisasi comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Accuracy comparison
results_df.set_index('Model')[['CV Mean', 'Test Accuracy']].plot(
    kind='bar', ax=axes[0], rot=0
)
axes[0].set_ylabel('Score')
axes[0].set_title('Accuracy Comparison')
axes[0].legend(['CV Mean', 'Test Accuracy'])
axes[0].set_ylim(0.7, 1.0)

# Precision, Recall, F1 comparison
results_df.set_index('Model')[['Precision', 'Recall', 'F1-Score']].plot(
    kind='bar', ax=axes[1], rot=0
)
axes[1].set_ylabel('Score')
axes[1].set_title('Classification Metrics Comparison')
axes[1].set_ylim(0.5, 1.0)

plt.tight_layout()
plt.show()
```

## 💪 Tugas Praktikum

### Tugas 1: Random Forest vs Decision Tree (25 poin)

Gunakan dataset classification (Iris, Wine, atau Breast Cancer):
1. Train single Decision Tree
2. Train Random Forest dengan n_estimators = [10, 50, 100, 200]
3. Bandingkan accuracy, precision, recall
4. Analisis feature importance
5. Plot learning curve untuk berbagai n_estimators
6. Kesimpulan: Berapa banyak trees yang optimal?

### Tugas 2: Gradient Boosting Tuning (30 poin)

Gunakan dataset yang sama:
1. Train Gradient Boosting dengan default parameters
2. Experiment dengan learning_rate: [0.01, 0.05, 0.1, 0.5, 1.0]
3. Untuk setiap learning rate, plot training progress (staged_predict)
4. Experiment dengan max_depth: [2, 3, 5, 7, 10]
5. Visualisasi hasil dengan heatmap atau line plots
6. Identifikasi best combination
7. Interpretasi: Bagaimana learning rate mempengaruhi convergence?

### Tugas 3: Regression dengan Ensemble (25 poin)

Gunakan California Housing atau dataset regression lain:
1. Train Decision Tree Regressor
2. Train Random Forest Regressor
3. Train Gradient Boosting Regressor
4. Bandingkan R² score dan RMSE
5. Analisis feature importance dari setiap model
6. Visualisasi predicted vs actual untuk ketiga model
7. Kesimpulan: Model mana yang terbaik? Mengapa?

### Tugas 4: Real-World Application - Customer Churn (20 poin)

Download dataset churn prediction dari Kaggle atau gunakan synthetic data:
1. EDA singkat (distribusi kelas, feature correlation)
2. Preprocessing (handling missing values, encoding)
3. Train 3 models: Decision Tree, Random Forest, Gradient Boosting
4. Gunakan cross-validation untuk evaluasi
5. Pilih model terbaik dan lakukan hyperparameter tuning
6. Final evaluation pada test set
7. Business recommendation: Feature mana yang paling penting untuk reduce churn?

## 📤 Cara Mengumpulkan

1. Notebook lengkap dengan semua tugas
2. Setiap section ada interpretasi dan analisis
3. Export ke PDF: `NIM_Nama_Pertemuan06.pdf`
4. Upload ke LMS atau GitHub

## ✅ Kriteria Penilaian

| Aspek | Bobot |
|-------|-------|
| Tugas 1: RF vs DT Comparison | 25% |
| Tugas 2: GB Tuning | 30% |
| Tugas 3: Regression Ensemble | 25% |
| Tugas 4: Real-World Application | 20% |
| Feature importance analysis | 15% |
| Interpretasi hasil tuning | 15% |
| Dokumentasi | 10% |

## 📚 Referensi

1. [Random Forest - Scikit-learn](https://scikit-learn.org/stable/modules/ensemble.html#forest)
2. [Gradient Boosting - Scikit-learn](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)
3. [Understanding Random Forests](https://towardsdatascience.com/understanding-random-forest-58381e0602d2)
4. [Gradient Boosting Explained](https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/)

## 💡 Tips

- **Random Forest**: Lebih banyak trees biasanya lebih baik, tapi diminishing returns setelah ~100-200 trees
- **Gradient Boosting**: Learning rate kecil + banyak estimators = slower but better
- **Feature Importance**: Bisa berbeda antara RF dan GB, karena cara kerjanya berbeda
- **Imbalanced data?** Gunakan `class_weight='balanced'` atau adjust dengan SMOTE
- **Regression?** RMSE lebih interpretable daripada MSE karena sama unit dengan target

---

**Happy Ensembling! 🌳🌲🌴**
