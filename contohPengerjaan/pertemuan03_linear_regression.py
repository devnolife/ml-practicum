"""
Pertemuan 3: Linear Regression
Contoh program lengkap untuk Simple Linear Regression, Multiple Regression, dan Polynomial Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print("=" * 60)
print("PERTEMUAN 3: Linear Regression")
print("=" * 60)

# ============================================================================
# 1. SIMPLE LINEAR REGRESSION (1 Variabel)
# ============================================================================
print("\n1. SIMPLE LINEAR REGRESSION")
print("-" * 60)

# Buat data: Jam belajar vs Nilai
np.random.seed(42)
jam_belajar = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
nilai = 50 + 5 * jam_belajar.squeeze() + np.random.randn(10) * 3

print("Data: Jam Belajar vs Nilai")
df_simple = pd.DataFrame({
    'Jam Belajar': jam_belajar.squeeze(),
    'Nilai': nilai
})
print(df_simple)

# Buat model
model_simple = LinearRegression()
model_simple.fit(jam_belajar, nilai)

# Prediksi
prediksi_simple = model_simple.predict(jam_belajar)

# Evaluasi model
r2_simple = r2_score(nilai, prediksi_simple)
rmse_simple = np.sqrt(mean_squared_error(nilai, prediksi_simple))

print(f"\nModel: y = {model_simple.intercept_:.2f} + {model_simple.coef_[0]:.2f}x")
print(f"R² Score: {r2_simple:.4f}")
print(f"RMSE: {rmse_simple:.4f}")

print("\nInterpretasi:")
print(f"- Intercept ({model_simple.intercept_:.2f}): Nilai dasar tanpa belajar")
print(f"- Slope ({model_simple.coef_[0]:.2f}): Setiap tambahan 1 jam belajar, nilai naik {model_simple.coef_[0]:.2f} poin")

# Prediksi untuk nilai baru
jam_baru = np.array([[12]])
nilai_prediksi = model_simple.predict(jam_baru)
print(f"\nPrediksi: Jika belajar {jam_baru[0][0]} jam, nilai diprediksi = {nilai_prediksi[0]:.2f}")

# Visualisasi
plt.figure(figsize=(10, 6))
plt.scatter(jam_belajar, nilai, color='blue', s=100, alpha=0.6, label='Data Asli')
plt.plot(jam_belajar, prediksi_simple, color='red', linewidth=2, label='Garis Regresi')
plt.xlabel('Jam Belajar', fontsize=12)
plt.ylabel('Nilai', fontsize=12)
plt.title('Simple Linear Regression: Jam Belajar vs Nilai', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.text(2, 75, f'y = {model_simple.intercept_:.2f} + {model_simple.coef_[0]:.2f}x\nR² = {r2_simple:.4f}', 
         fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.tight_layout()
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan03_simple_regression.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualisasi Simple Linear Regression disimpan")
plt.close()

# ============================================================================
# 2. MULTIPLE LINEAR REGRESSION (Banyak Variabel)
# ============================================================================
print("\n2. MULTIPLE LINEAR REGRESSION")
print("-" * 60)

# Buat data: Luas Rumah, Jumlah Kamar, Umur Bangunan vs Harga
np.random.seed(42)
n_samples = 100

luas = np.random.randint(30, 200, n_samples)  # m²
kamar = np.random.randint(1, 6, n_samples)    # jumlah kamar
umur = np.random.randint(0, 30, n_samples)    # tahun

# Harga = 100 + 2*luas + 50*kamar - 1*umur + noise
harga = 100 + 2*luas + 50*kamar - 1*umur + np.random.randn(n_samples) * 20

# DataFrame
df_multiple = pd.DataFrame({
    'Luas': luas,
    'Kamar': kamar,
    'Umur': umur,
    'Harga': harga
})

print("Data: Real Estate (5 baris pertama)")
print(df_multiple.head())

print("\nStatistik Deskriptif:")
print(df_multiple.describe())

# Split features dan target
X_multiple = df_multiple[['Luas', 'Kamar', 'Umur']]
y_multiple = df_multiple['Harga']

# Split train dan test
X_train, X_test, y_train, y_test = train_test_split(X_multiple, y_multiple, 
                                                      test_size=0.2, random_state=42)

print(f"\nData Training: {len(X_train)} samples")
print(f"Data Testing: {len(X_test)} samples")

# Buat dan latih model
model_multiple = LinearRegression()
model_multiple.fit(X_train, y_train)

# Prediksi
y_train_pred = model_multiple.predict(X_train)
y_test_pred = model_multiple.predict(X_test)

# Evaluasi
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae_test = mean_absolute_error(y_test, y_test_pred)

print("\n--- Evaluasi Model ---")
print(f"Training Set:")
print(f"  R² Score: {r2_train:.4f}")
print(f"  RMSE: {rmse_train:.4f}")

print(f"\nTesting Set:")
print(f"  R² Score: {r2_test:.4f}")
print(f"  RMSE: {rmse_test:.4f}")
print(f"  MAE: {mae_test:.4f}")

print("\n--- Koefisien Model ---")
print(f"Intercept: {model_multiple.intercept_:.2f}")
for feature, coef in zip(X_multiple.columns, model_multiple.coef_):
    print(f"{feature}: {coef:.2f}")

print("\nInterpretasi:")
print(f"- Setiap tambahan 1 m² luas, harga naik {model_multiple.coef_[0]:.2f} juta")
print(f"- Setiap tambahan 1 kamar, harga naik {model_multiple.coef_[1]:.2f} juta")
print(f"- Setiap tambahan 1 tahun umur, harga turun {abs(model_multiple.coef_[2]):.2f} juta")

# Prediksi contoh baru
rumah_baru = pd.DataFrame({
    'Luas': [100],
    'Kamar': [3],
    'Umur': [5]
})
harga_prediksi = model_multiple.predict(rumah_baru)
print(f"\nPrediksi: Rumah dengan luas 100m², 3 kamar, umur 5 tahun")
print(f"Harga diprediksi = Rp {harga_prediksi[0]:.2f} juta")

# Visualisasi: Actual vs Predicted
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Multiple Linear Regression - Real Estate', fontsize=16, fontweight='bold')

# Plot 1: Training Set
axes[0].scatter(y_train, y_train_pred, alpha=0.6, color='blue', edgecolors='black')
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Harga Actual', fontsize=12)
axes[0].set_ylabel('Harga Predicted', fontsize=12)
axes[0].set_title(f'Training Set (R² = {r2_train:.4f})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Testing Set
axes[1].scatter(y_test, y_test_pred, alpha=0.6, color='green', edgecolors='black')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Harga Actual', fontsize=12)
axes[1].set_ylabel('Harga Predicted', fontsize=12)
axes[1].set_title(f'Testing Set (R² = {r2_test:.4f})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan03_multiple_regression.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualisasi Multiple Linear Regression disimpan")
plt.close()

# Residual plot
plt.figure(figsize=(10, 6))
residuals = y_test - y_test_pred
plt.scatter(y_test_pred, residuals, alpha=0.6, color='purple', edgecolors='black')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Predicted Values', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Residual Plot', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan03_residual_plot.png', dpi=300, bbox_inches='tight')
print("✓ Residual Plot disimpan")
plt.close()

# ============================================================================
# 3. POLYNOMIAL REGRESSION (Data Non-Linear)
# ============================================================================
print("\n3. POLYNOMIAL REGRESSION")
print("-" * 60)

# Buat data non-linear (melengkung)
np.random.seed(42)
X_poly = np.linspace(0, 10, 50).reshape(-1, 1)
y_poly = 2 + 3*X_poly.squeeze() - 0.5*X_poly.squeeze()**2 + np.random.randn(50) * 2

print("Data: Non-linear Relationship")

# Linear Regression (untuk perbandingan)
model_linear_poly = LinearRegression()
model_linear_poly.fit(X_poly, y_poly)
y_pred_linear = model_linear_poly.predict(X_poly)
r2_linear = r2_score(y_poly, y_pred_linear)

# Polynomial Regression (degree 2)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly_features = poly_features.fit_transform(X_poly)

model_poly = LinearRegression()
model_poly.fit(X_poly_features, y_poly)
y_pred_poly = model_poly.predict(X_poly_features)
r2_poly = r2_score(y_poly, y_pred_poly)

print(f"\nLinear Regression R²: {r2_linear:.4f}")
print(f"Polynomial Regression R²: {r2_poly:.4f}")
print(f"\nPerbandingan: Polynomial lebih baik {(r2_poly - r2_linear)*100:.2f}% dari Linear")

# Visualisasi
plt.figure(figsize=(12, 6))
plt.scatter(X_poly, y_poly, color='blue', s=50, alpha=0.6, label='Data Asli')
plt.plot(X_poly, y_pred_linear, color='red', linewidth=2, label=f'Linear (R² = {r2_linear:.4f})')
plt.plot(X_poly, y_pred_poly, color='green', linewidth=2, label=f'Polynomial Degree 2 (R² = {r2_poly:.4f})')
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('Polynomial Regression vs Linear Regression', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan03_polynomial_regression.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualisasi Polynomial Regression disimpan")
plt.close()

# Compare different polynomial degrees
degrees = [1, 2, 3, 4, 5]
plt.figure(figsize=(14, 8))

for i, degree in enumerate(degrees, 1):
    plt.subplot(2, 3, i)
    
    if degree == 1:
        # Linear
        model = LinearRegression()
        model.fit(X_poly, y_poly)
        y_pred = model.predict(X_poly)
    else:
        # Polynomial
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly_deg = poly.fit_transform(X_poly)
        model = LinearRegression()
        model.fit(X_poly_deg, y_poly)
        y_pred = model.predict(X_poly_deg)
    
    r2 = r2_score(y_poly, y_pred)
    
    plt.scatter(X_poly, y_poly, color='blue', s=30, alpha=0.5)
    plt.plot(X_poly, y_pred, color='red', linewidth=2)
    plt.title(f'Degree {degree} (R² = {r2:.4f})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)

plt.suptitle('Polynomial Regression - Different Degrees', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan03_polynomial_degrees.png', dpi=300, bbox_inches='tight')
print("✓ Visualisasi berbagai derajat polynomial disimpan")
plt.close()

print("\n" + "=" * 60)
print("SELESAI - Pertemuan 3")
print("=" * 60)
print("\nRingkasan yang dipelajari:")
print("✓ Simple Linear Regression (1 variabel)")
print("✓ Multiple Linear Regression (banyak variabel)")
print("✓ Polynomial Regression (data non-linear)")
print("✓ Model evaluation (R², RMSE, MAE)")
print("✓ Train-test split")
print("✓ Residual analysis")
print("\nFile yang dibuat:")
print("- pertemuan03_simple_regression.png")
print("- pertemuan03_multiple_regression.png")
print("- pertemuan03_residual_plot.png")
print("- pertemuan03_polynomial_regression.png")
print("- pertemuan03_polynomial_degrees.png")
