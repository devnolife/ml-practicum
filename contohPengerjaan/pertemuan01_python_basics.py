"""
Pertemuan 1: Pengenalan Python untuk Machine Learning
Contoh program lengkap untuk NumPy, Pandas, dan Matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 60)
print("PERTEMUAN 1: Python Basics untuk Machine Learning")
print("=" * 60)

# ============================================================================
# 1. NUMPY - Untuk Operasi Numerik
# ============================================================================
print("\n1. NUMPY - Operasi Array dan Matrix")
print("-" * 60)

# Buat array sederhana
angka = np.array([10, 20, 30, 40, 50])
print(f"Array: {angka}")
print(f"Rata-rata: {angka.mean()}")
print(f"Total: {angka.sum()}")
print(f"Nilai Max: {angka.max()}")
print(f"Nilai Min: {angka.min()}")

# Buat matrix 2D
matrix = np.array([[1, 2, 3], 
                   [4, 5, 6], 
                   [7, 8, 9]])
print(f"\nMatrix 2D:\n{matrix}")
print(f"Bentuk matrix: {matrix.shape}")
print(f"Rata-rata setiap kolom: {matrix.mean(axis=0)}")
print(f"Rata-rata setiap baris: {matrix.mean(axis=1)}")

# Operasi matematika
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([10, 20, 30, 40, 50])
print(f"\nArray 1: {array1}")
print(f"Array 2: {array2}")
print(f"Penjumlahan: {array1 + array2}")
print(f"Perkalian: {array1 * array2}")
print(f"Pangkat 2: {array1 ** 2}")

# Generate array dengan pola
print("\nGenerate array dengan pola:")
print(f"Angka 0-9: {np.arange(10)}")
print(f"Angka 0-20 dengan step 2: {np.arange(0, 21, 2)}")
print(f"10 angka antara 0-1: {np.linspace(0, 1, 10)}")

# ============================================================================
# 2. PANDAS - Untuk Manipulasi Data
# ============================================================================
print("\n2. PANDAS - Manipulasi Data")
print("-" * 60)

# Buat DataFrame sederhana
data = {
    'Nama': ['Budi', 'Ani', 'Citra', 'Doni', 'Eka'],
    'Umur': [23, 21, 22, 24, 23],
    'Nilai': [85, 90, 78, 88, 92],
    'Kota': ['Jakarta', 'Bandung', 'Jakarta', 'Surabaya', 'Bandung']
}
df = pd.DataFrame(data)

print("\nDataFrame Mahasiswa:")
print(df)

# Statistik deskriptif
print("\nStatistik Deskriptif:")
print(df.describe())

# Filter data
print("\nMahasiswa dengan nilai >= 85:")
print(df[df['Nilai'] >= 85])

print("\nMahasiswa dari Jakarta:")
print(df[df['Kota'] == 'Jakarta'])

# Group by dan agregasi
print("\nRata-rata nilai per kota:")
print(df.groupby('Kota')['Nilai'].mean())

print("\nJumlah mahasiswa per kota:")
print(df['Kota'].value_counts())

# Buat data untuk latihan lebih lanjut
data_penjualan = {
    'Bulan': ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun'],
    'Penjualan': [100, 120, 115, 130, 140, 135],
    'Biaya': [80, 90, 85, 95, 100, 95],
    'Profit': [20, 30, 30, 35, 40, 40]
}
df_sales = pd.DataFrame(data_penjualan)

print("\nData Penjualan:")
print(df_sales)

# Hitung total dan rata-rata
print(f"\nTotal Penjualan: {df_sales['Penjualan'].sum()}")
print(f"Rata-rata Penjualan: {df_sales['Penjualan'].mean():.2f}")
print(f"Total Profit: {df_sales['Profit'].sum()}")

# ============================================================================
# 3. MATPLOTLIB - Untuk Visualisasi Data
# ============================================================================
print("\n3. MATPLOTLIB - Visualisasi Data")
print("-" * 60)
print("Membuat visualisasi... (close plot window to continue)")

# Buat figure dengan multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Pertemuan 1: Contoh Visualisasi Data', fontsize=16, fontweight='bold')

# Plot 1: Line Plot - Trend Penjualan
axes[0, 0].plot(df_sales['Bulan'], df_sales['Penjualan'], 
                marker='o', linewidth=2, markersize=8, color='blue')
axes[0, 0].set_title('Trend Penjualan per Bulan')
axes[0, 0].set_xlabel('Bulan')
axes[0, 0].set_ylabel('Penjualan (juta)')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Bar Plot - Profit per Bulan
axes[0, 1].bar(df_sales['Bulan'], df_sales['Profit'], color='green', alpha=0.7)
axes[0, 1].set_title('Profit per Bulan')
axes[0, 1].set_xlabel('Bulan')
axes[0, 1].set_ylabel('Profit (juta)')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3: Scatter Plot - Penjualan vs Profit
axes[1, 0].scatter(df_sales['Penjualan'], df_sales['Profit'], 
                   s=100, c='red', alpha=0.6, edgecolors='black')
axes[1, 0].set_title('Hubungan Penjualan vs Profit')
axes[1, 0].set_xlabel('Penjualan (juta)')
axes[1, 0].set_ylabel('Profit (juta)')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Multiple Lines - Penjualan, Biaya, Profit
axes[1, 1].plot(df_sales['Bulan'], df_sales['Penjualan'], 
                marker='o', label='Penjualan', linewidth=2)
axes[1, 1].plot(df_sales['Bulan'], df_sales['Biaya'], 
                marker='s', label='Biaya', linewidth=2)
axes[1, 1].plot(df_sales['Bulan'], df_sales['Profit'], 
                marker='^', label='Profit', linewidth=2)
axes[1, 1].set_title('Perbandingan Penjualan, Biaya, dan Profit')
axes[1, 1].set_xlabel('Bulan')
axes[1, 1].set_ylabel('Nilai (juta)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan01_visualisasi.png', dpi=300, bbox_inches='tight')
print("✓ Visualisasi disimpan sebagai 'pertemuan01_visualisasi.png'")
plt.show()

# Membuat plot tambahan untuk histogram
plt.figure(figsize=(10, 6))
nilai_mahasiswa = np.random.normal(75, 10, 100)  # Generate 100 nilai dengan mean=75, std=10
plt.hist(nilai_mahasiswa, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Distribusi Nilai Mahasiswa', fontsize=14, fontweight='bold')
plt.xlabel('Nilai')
plt.ylabel('Frekuensi')
plt.axvline(nilai_mahasiswa.mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {nilai_mahasiswa.mean():.2f}')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan01_histogram.png', dpi=300, bbox_inches='tight')
print("✓ Histogram disimpan sebagai 'pertemuan01_histogram.png'")
plt.show()

print("\n" + "=" * 60)
print("SELESAI - Pertemuan 1")
print("=" * 60)
print("\nRingkasan yang dipelajari:")
print("✓ NumPy: Array operations, mathematical functions")
print("✓ Pandas: DataFrame, filtering, grouping, aggregation")
print("✓ Matplotlib: Line plots, bar charts, scatter plots, histograms")
print("\nFile yang dibuat:")
print("- pertemuan01_visualisasi.png")
print("- pertemuan01_histogram.png")
