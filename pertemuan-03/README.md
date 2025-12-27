# Pertemuan 3: Linear Regression dan Polynomial Regression

## 🎯 Tujuan Pembelajaran

Setelah menyelesaikan pertemuan ini, mahasiswa diharapkan mampu:
1. Memahami konsep dan teori Linear Regression
2. Mengimplementasikan Linear Regression dengan scikit-learn
3. Memahami perbedaan Simple vs Multiple Linear Regression
4. Mengimplementasikan Polynomial Regression
5. Melakukan evaluasi model regresi dengan metrik yang tepat
6. Memahami konsep overfitting dan underfitting

## 📚 Teori Singkat

### Linear Regression

Linear Regression adalah algoritma supervised learning untuk memprediksi nilai kontinu (continuous value). Model ini mencari hubungan linear antara fitur (X) dan target (y).

**Formula:**
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε

di mana:
- y: target variable (yang diprediksi)
- β₀: intercept
- β₁, β₂, ..., βₙ: coefficients
- x₁, x₂, ..., xₙ: features
- ε: error term
```

**Simple Linear Regression**: 1 fitur (x)
```
y = β₀ + β₁x
```

**Multiple Linear Regression**: > 1 fitur
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

### Polynomial Regression

Polynomial Regression adalah ekstensi dari Linear Regression yang dapat menangkap hubungan non-linear dengan menambahkan polynomial terms.

**Formula (degree 2):**
```
y = β₀ + β₁x + β₂x²
```

### Metrics Evaluasi

1. **MAE (Mean Absolute Error)**: Rata-rata absolute error
   ```
   MAE = (1/n) Σ|yᵢ - ŷᵢ|
   ```

2. **MSE (Mean Squared Error)**: Rata-rata squared error
   ```
   MSE = (1/n) Σ(yᵢ - ŷᵢ)²
   ```

3. **RMSE (Root Mean Squared Error)**: Akar dari MSE
   ```
   RMSE = √MSE
   ```

4. **R² Score**: Proporsi variance yang dijelaskan model (0-1, semakin tinggi semakin baik)
   ```
   R² = 1 - (SS_res / SS_tot)
   ```

## 📝 Praktikum

### Persiapan: Import Library

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Setting
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
```

### Langkah 1: Simple Linear Regression

```python
# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # 100 samples, 1 feature
y = 2.5 * X.squeeze() + 5 + np.random.randn(100) * 2  # y = 2.5x + 5 + noise

# Visualisasi data
plt.scatter(X, y, alpha=0.6)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Dataset untuk Simple Linear Regression')
plt.show()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediksi
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Koefisien
print(f"Intercept (β₀): {model.intercept_:.2f}")
print(f"Coefficient (β₁): {model.coef_[0]:.2f}")
print(f"\nModel equation: y = {model.intercept_:.2f} + {model.coef_[0]:.2f}x")

# Visualisasi hasil
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, alpha=0.6, label='Training data')
plt.plot(X_train, y_pred_train, color='red', linewidth=2, label='Regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Training Set')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, alpha=0.6, label='Testing data', color='green')
plt.plot(X_test, y_pred_test, color='red', linewidth=2, label='Regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Testing Set')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluasi
print("\n=== Evaluasi Model ===")
print(f"Train R² Score: {r2_score(y_train, y_pred_train):.4f}")
print(f"Test R² Score: {r2_score(y_test, y_pred_test):.4f}")
print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")
```

### Langkah 2: Multiple Linear Regression - Prediksi Harga Rumah

```python
# Load dataset California Housing
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target

print("=== Info Dataset ===")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nFeatures: {housing.feature_names}")
print(f"\nTarget: Median house value (in $100k)")

# Exploratory Data Analysis
print("\n=== Statistik Deskriptif ===")
print(df.describe())

# Correlation dengan target
correlations = df.corr()['Price'].sort_values(ascending=False)
print("\n=== Korelasi dengan Price ===")
print(correlations)

# Visualisasi correlation
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Prepare data
X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

# Prediksi
y_pred_train = model_multi.predict(X_train)
y_pred_test = model_multi.predict(X_test)

# Koefisien
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model_multi.coef_
}).sort_values('Coefficient', ascending=False)

print("\n=== Koefisien Model ===")
print(coef_df)
print(f"\nIntercept: {model_multi.intercept_:.4f}")

# Visualisasi koefisien
plt.figure(figsize=(10, 6))
plt.barh(coef_df['Feature'], coef_df['Coefficient'])
plt.xlabel('Coefficient Value')
plt.title('Feature Coefficients')
plt.tight_layout()
plt.show()

# Evaluasi
print("\n=== Evaluasi Model ===")
print(f"Train R² Score: {r2_score(y_train, y_pred_train):.4f}")
print(f"Test R² Score: {r2_score(y_test, y_pred_test):.4f}")
print(f"Train MAE: {mean_absolute_error(y_train, y_pred_train):.4f}")
print(f"Test MAE: {mean_absolute_error(y_test, y_pred_test):.4f}")
print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")

# Visualisasi Predicted vs Actual
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train, alpha=0.3)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Training Set: Predicted vs Actual')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, alpha=0.3, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Testing Set: Predicted vs Actual')

plt.tight_layout()
plt.show()

# Residual Analysis
residuals_train = y_train - y_pred_train
residuals_test = y_test - y_pred_test

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_pred_train, residuals_train, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residual Plot - Training Set')

plt.subplot(1, 2, 2)
plt.hist(residuals_test, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residual Distribution - Test Set')

plt.tight_layout()
plt.show()
```

### Langkah 3: Polynomial Regression

```python
# Generate non-linear data
np.random.seed(42)
X_poly = np.linspace(-3, 3, 100).reshape(-1, 1)
y_poly = 0.5 * X_poly**2 + X_poly + 2 + np.random.randn(100, 1) * 0.5

# Visualisasi data
plt.scatter(X_poly, y_poly, alpha=0.6)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Non-linear Dataset')
plt.show()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y_poly, test_size=0.2, random_state=42
)

# Compare different polynomial degrees
degrees = [1, 2, 3, 5, 10]
plt.figure(figsize=(15, 10))

for idx, degree in enumerate(degrees, 1):
    # Transform features
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    
    # Train model
    model_poly = LinearRegression()
    model_poly.fit(X_train_poly, y_train)
    
    # Predict
    y_pred_train = model_poly.predict(X_train_poly)
    y_pred_test = model_poly.predict(X_test_poly)
    
    # For plotting
    X_range = np.linspace(-3, 3, 300).reshape(-1, 1)
    X_range_poly = poly_features.transform(X_range)
    y_range_pred = model_poly.predict(X_range_poly)
    
    # Plot
    plt.subplot(2, 3, idx)
    plt.scatter(X_train, y_train, alpha=0.6, label='Train')
    plt.scatter(X_test, y_test, alpha=0.6, color='green', label='Test')
    plt.plot(X_range, y_range_pred, color='red', linewidth=2, label=f'Degree {degree}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Polynomial Degree {degree}\nTrain R²: {r2_score(y_train, y_pred_train):.3f}, Test R²: {r2_score(y_test, y_pred_test):.3f}')
    plt.legend()
    plt.ylim(-5, 10)

plt.tight_layout()
plt.show()

# Detailed evaluation for degree 2 (best fit)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

model_best = LinearRegression()
model_best.fit(X_train_poly, y_train)

y_pred_train = model_best.predict(X_train_poly)
y_pred_test = model_best.predict(X_test_poly)

print("=== Best Model (Degree 2) ===")
print(f"Train R² Score: {r2_score(y_train, y_pred_train):.4f}")
print(f"Test R² Score: {r2_score(y_test, y_pred_test):.4f}")
print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")
```

### Langkah 4: Overfitting vs Underfitting Analysis

```python
# Training curve untuk berbagai degree
degrees = range(1, 15)
train_scores = []
test_scores = []

X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y_poly, test_size=0.2, random_state=42
)

for degree in degrees:
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    train_scores.append(r2_score(y_train, model.predict(X_train_poly)))
    test_scores.append(r2_score(y_test, model.predict(X_test_poly)))

# Visualisasi learning curve
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_scores, 'o-', label='Training Score', linewidth=2)
plt.plot(degrees, test_scores, 'o-', label='Testing Score', linewidth=2)
plt.xlabel('Polynomial Degree')
plt.ylabel('R² Score')
plt.title('Model Complexity vs Performance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axvline(x=2, color='r', linestyle='--', label='Optimal degree')
plt.tight_layout()
plt.show()

print("=== Analysis ===")
print("Degree 1: Underfitting (too simple)")
print("Degree 2-3: Good fit (optimal)")
print("Degree 10+: Overfitting (too complex)")
```

## 💪 Tugas Praktikum

### Tugas 1: Simple Linear Regression (20 poin)

Buat dataset sendiri atau gunakan dataset sederhana:
1. Buat scatter plot data
2. Implementasikan Simple Linear Regression
3. Visualisasi regression line
4. Hitung dan interpretasikan R² score, MAE, dan RMSE
5. Buat prediksi untuk 5 nilai baru

### Tugas 2: Multiple Linear Regression - Boston Housing (35 poin)

Gunakan dataset Boston Housing (atau alternatif: Ames Housing dari Kaggle):
1. Load dan eksplorasi dataset
2. Lakukan feature selection berdasarkan correlation
3. Split data (80-20)
4. Train model dengan semua fitur
5. Train model dengan top 5 fitur (based on correlation)
6. Bandingkan performa kedua model
7. Analisis residuals
8. Interpretasikan koefisien model

### Tugas 3: Polynomial Regression (25 poin)

Buat atau gunakan dataset non-linear:
1. Visualisasi data
2. Coba polynomial degree 1 sampai 10
3. Plot learning curve (train vs test score)
4. Identifikasi degree optimal
5. Jelaskan kapan terjadi underfitting dan overfitting

### Tugas 4: Real-World Application (20 poin)

Pilih salah satu kasus:
- Prediksi gaji berdasarkan years of experience
- Prediksi konsumsi bahan bakar kendaraan
- Prediksi harga laptop berdasarkan spesifikasi

Lakukan:
1. Collect atau download dataset
2. EDA singkat
3. Preprocessing (handling missing values, scaling jika perlu)
4. Train model (linear atau polynomial)
5. Evaluasi dan interpretasi hasil
6. Buat kesimpulan: apakah model cukup baik? Apa yang bisa ditingkatkan?

## 📤 Cara Mengumpulkan

1. Satu notebook untuk semua tugas dengan section yang jelas
2. Setiap tugas harus ada penjelasan dan interpretasi
3. Export ke PDF: `NIM_Nama_Pertemuan03.pdf`
4. Upload ke LMS atau GitHub

## ✅ Kriteria Penilaian

| Aspek | Bobot |
|-------|-------|
| Tugas 1: Simple Linear Regression | 20% |
| Tugas 2: Multiple Linear Regression | 35% |
| Tugas 3: Polynomial Regression | 25% |
| Tugas 4: Real-World Application | 20% |
| Interpretasi hasil | 20% |
| Kualitas visualisasi | 15% |
| Dokumentasi | 10% |

## 📚 Referensi

1. [Scikit-learn Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
2. [Understanding Polynomial Regression](https://towardsdatascience.com/polynomial-regression-bbe8b9d97491)
3. [Regression Metrics Explained](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)

## 💡 Tips

- R² > 0.7 biasanya dianggap model yang baik
- RMSE memberikan gambaran error dalam unit asli
- Jangan lupa cek residual plot untuk validasi asumsi linear regression
- Polynomial degree terlalu tinggi = overfitting!

---

**Happy Modeling! 📈🏠**
