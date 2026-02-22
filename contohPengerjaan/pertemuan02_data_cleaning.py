"""
Pertemuan 2: Data Cleaning dan Exploratory Data Analysis (EDA)
Contoh program lengkap untuk cleaning data, handling missing values, outliers, dan scaling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

print("=" * 60)
print("PERTEMUAN 2: Data Cleaning & EDA")
print("=" * 60)

# ============================================================================
# 1. LOAD DATA DAN INSPEKSI AWAL
# ============================================================================
print("\n1. LOAD DATA DAN INSPEKSI")
print("-" * 60)

# Load dataset Titanic
df = sns.load_dataset('titanic')

print("Data Titanic berhasil dimuat!")
print(f"\nJumlah baris: {len(df)}")
print(f"Jumlah kolom: {len(df.columns)}")

print("\n5 Baris Pertama:")
print(df.head())

print("\nInformasi Dataset:")
print(df.info())

print("\nStatistik Deskriptif:")
print(df.describe())

# ============================================================================
# 2. IDENTIFIKASI MISSING VALUES
# ============================================================================
print("\n2. IDENTIFIKASI MISSING VALUES")
print("-" * 60)

missing = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100

missing_df = pd.DataFrame({
    'Kolom': missing.index,
    'Jumlah Missing': missing.values,
    'Persentase (%)': missing_percent.values
})
missing_df = missing_df[missing_df['Jumlah Missing'] > 0].sort_values('Jumlah Missing', ascending=False)

print("\nKolom dengan Missing Values:")
print(missing_df.to_string(index=False))

# Visualisasi missing values
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
plt.title('Visualisasi Missing Values (Kuning = Missing)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan02_missing_values.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualisasi missing values disimpan")
plt.close()

# ============================================================================
# 3. HANDLING MISSING VALUES
# ============================================================================
print("\n3. HANDLING MISSING VALUES")
print("-" * 60)

# Copy dataframe untuk cleaning
df_clean = df.copy()

# Isi age dengan median (robust terhadap outliers)
age_median = df_clean['age'].median()
df_clean['age'].fillna(age_median, inplace=True)
print(f"✓ Kolom 'age': Missing values diisi dengan median = {age_median:.1f}")

# Isi embark_town dengan mode (nilai paling sering)
embark_mode = df_clean['embark_town'].mode()[0]
df_clean['embark_town'].fillna(embark_mode, inplace=True)
print(f"✓ Kolom 'embark_town': Missing values diisi dengan mode = {embark_mode}")

# Drop kolom deck (terlalu banyak missing values)
df_clean.drop('deck', axis=1, inplace=True)
print("✓ Kolom 'deck': Dihapus (terlalu banyak missing values)")

# Verifikasi
print(f"\nMissing values setelah cleaning: {df_clean.isnull().sum().sum()}")

# ============================================================================
# 4. HANDLING DUPLICATES
# ============================================================================
print("\n4. HANDLING DUPLICATES")
print("-" * 60)

duplicates_before = df_clean.duplicated().sum()
print(f"Jumlah data duplikat: {duplicates_before}")

if duplicates_before > 0:
    df_clean.drop_duplicates(inplace=True)
    print(f"✓ {duplicates_before} data duplikat telah dihapus")
else:
    print("✓ Tidak ada data duplikat")

print(f"Jumlah baris setelah cleaning: {len(df_clean)}")

# ============================================================================
# 5. DETEKSI DAN HANDLING OUTLIERS
# ============================================================================
print("\n5. DETEKSI DAN HANDLING OUTLIERS")
print("-" * 60)

# Fokus pada kolom numerik: age, fare
numeric_cols = ['age', 'fare']

# Visualisasi outliers dengan boxplot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Deteksi Outliers dengan Boxplot', fontsize=14, fontweight='bold')

for idx, col in enumerate(numeric_cols):
    axes[idx].boxplot(df_clean[col].dropna())
    axes[idx].set_title(f'Boxplot: {col.capitalize()}')
    axes[idx].set_ylabel('Value')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan02_outliers_boxplot.png', dpi=300, bbox_inches='tight')
print("✓ Boxplot outliers disimpan")
plt.close()

# Deteksi outliers dengan IQR method
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

print("\nDeteksi Outliers dengan IQR Method:")
for col in numeric_cols:
    outliers, lower, upper = detect_outliers_iqr(df_clean, col)
    print(f"  {col}: {len(outliers)} outliers detected (Range: {lower:.2f} - {upper:.2f})")

# Handle outliers: cap dengan upper/lower bound (winsorization)
df_clean_no_outliers = df_clean.copy()
for col in numeric_cols:
    Q1 = df_clean_no_outliers[col].quantile(0.25)
    Q3 = df_clean_no_outliers[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_clean_no_outliers[col] = df_clean_no_outliers[col].clip(lower=lower_bound, upper=upper_bound)

print("\n✓ Outliers telah ditangani dengan winsorization")

# ============================================================================
# 6. FEATURE SCALING
# ============================================================================
print("\n6. FEATURE SCALING")
print("-" * 60)

# Pilih kolom numerik untuk scaling
scaling_cols = ['age', 'fare']
data_for_scaling = df_clean_no_outliers[scaling_cols].copy()

print("\nData SEBELUM Scaling:")
print(data_for_scaling.head())
print("\nStatistik SEBELUM Scaling:")
print(data_for_scaling.describe())

# StandardScaler (mean=0, std=1)
scaler_standard = StandardScaler()
data_standard_scaled = scaler_standard.fit_transform(data_for_scaling)
df_standard = pd.DataFrame(data_standard_scaled, columns=[f'{col}_standard' for col in scaling_cols])

print("\n--- StandardScaler (Standardization) ---")
print("Rumus: (x - mean) / std")
print(df_standard.head())
print(f"Mean: {df_standard.mean().values}")
print(f"Std: {df_standard.std().values}")

# MinMaxScaler (range 0-1)
scaler_minmax = MinMaxScaler()
data_minmax_scaled = scaler_minmax.fit_transform(data_for_scaling)
df_minmax = pd.DataFrame(data_minmax_scaled, columns=[f'{col}_minmax' for col in scaling_cols])

print("\n--- MinMaxScaler (Normalization) ---")
print("Rumus: (x - min) / (max - min)")
print(df_minmax.head())
print(f"Min: {df_minmax.min().values}")
print(f"Max: {df_minmax.max().values}")

# Visualisasi perbandingan scaling
fig, axes = plt.subplots(3, 2, figsize=(12, 12))
fig.suptitle('Perbandingan Feature Scaling', fontsize=16, fontweight='bold')

for idx, col in enumerate(scaling_cols):
    # Original
    axes[0, idx].hist(data_for_scaling[col], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, idx].set_title(f'Original: {col.capitalize()}')
    axes[0, idx].set_ylabel('Frequency')
    axes[0, idx].grid(True, alpha=0.3)
    
    # StandardScaler
    axes[1, idx].hist(df_standard[f'{col}_standard'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1, idx].set_title(f'StandardScaler: {col.capitalize()}')
    axes[1, idx].set_ylabel('Frequency')
    axes[1, idx].grid(True, alpha=0.3)
    
    # MinMaxScaler
    axes[2, idx].hist(df_minmax[f'{col}_minmax'], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[2, idx].set_title(f'MinMaxScaler: {col.capitalize()}')
    axes[2, idx].set_ylabel('Frequency')
    axes[2, idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan02_scaling_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualisasi perbandingan scaling disimpan")
plt.close()

# ============================================================================
# 7. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n7. EXPLORATORY DATA ANALYSIS")
print("-" * 60)

# Analisis survival rate
survival_rate = df_clean['survived'].value_counts()
survival_percent = (df_clean['survived'].value_counts(normalize=True) * 100).round(2)

print("\nSurvival Rate:")
print(f"  Survived (1): {survival_rate[1]} orang ({survival_percent[1]}%)")
print(f"  Not Survived (0): {survival_rate[0]} orang ({survival_percent[0]}%)")

# Survival by gender
print("\nSurvival Rate by Gender:")
survival_by_sex = df_clean.groupby('sex')['survived'].agg(['sum', 'count', 'mean'])
survival_by_sex['percentage'] = (survival_by_sex['mean'] * 100).round(2)
print(survival_by_sex)

# Survival by class
print("\nSurvival Rate by Class:")
survival_by_class = df_clean.groupby('pclass')['survived'].agg(['sum', 'count', 'mean'])
survival_by_class['percentage'] = (survival_by_class['mean'] * 100).round(2)
print(survival_by_class)

# Visualisasi EDA
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Exploratory Data Analysis - Titanic', fontsize=16, fontweight='bold')

# Plot 1: Survival count
survival_counts = df_clean['survived'].value_counts()
axes[0, 0].bar(['Not Survived', 'Survived'], survival_counts.values, color=['#FF6B6B', '#4ECDC4'])
axes[0, 0].set_title('Survival Count')
axes[0, 0].set_ylabel('Count')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Plot 2: Survival by Gender
survival_sex = df_clean.groupby(['sex', 'survived']).size().unstack()
survival_sex.plot(kind='bar', ax=axes[0, 1], color=['#FF6B6B', '#4ECDC4'])
axes[0, 1].set_title('Survival by Gender')
axes[0, 1].set_xlabel('Gender')
axes[0, 1].set_ylabel('Count')
axes[0, 1].legend(['Not Survived', 'Survived'])
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=0)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Plot 3: Survival by Class
survival_class = df_clean.groupby(['pclass', 'survived']).size().unstack()
survival_class.plot(kind='bar', ax=axes[1, 0], color=['#FF6B6B', '#4ECDC4'])
axes[1, 0].set_title('Survival by Class')
axes[1, 0].set_xlabel('Class')
axes[1, 0].set_ylabel('Count')
axes[1, 0].legend(['Not Survived', 'Survived'])
axes[1, 0].set_xticklabels(['1st Class', '2nd Class', '3rd Class'], rotation=0)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Age distribution by Survival
axes[1, 1].hist([df_clean[df_clean['survived']==0]['age'], 
                 df_clean[df_clean['survived']==1]['age']], 
                bins=20, label=['Not Survived', 'Survived'], 
                color=['#FF6B6B', '#4ECDC4'], alpha=0.7)
axes[1, 1].set_title('Age Distribution by Survival')
axes[1, 1].set_xlabel('Age')
axes[1, 1].set_ylabel('Count')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan02_eda_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualisasi EDA disimpan")
plt.close()

# Correlation heatmap
plt.figure(figsize=(10, 8))
numeric_df = df_clean.select_dtypes(include=[np.number])
correlation = numeric_df.corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan02_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Correlation heatmap disimpan")
plt.close()

# ============================================================================
# 8. SAVE CLEANED DATA
# ============================================================================
print("\n8. SAVE DATA YANG SUDAH DIBERSIHKAN")
print("-" * 60)

# Save to CSV
df_clean_no_outliers.to_csv('/workspaces/ml-practicum/contohPengerjaan/titanic_cleaned.csv', index=False)
print("✓ Data telah disimpan ke 'titanic_cleaned.csv'")

print("\n" + "=" * 60)
print("SELESAI - Pertemuan 2")
print("=" * 60)
print("\nRingkasan yang dipelajari:")
print("✓ Load dan inspeksi data")
print("✓ Identifikasi missing values")
print("✓ Handling missing values (median, mode, drop)")
print("✓ Deteksi dan handling outliers (IQR method)")
print("✓ Feature scaling (StandardScaler, MinMaxScaler)")
print("✓ Exploratory Data Analysis (EDA)")
print("\nFile yang dibuat:")
print("- pertemuan02_missing_values.png")
print("- pertemuan02_outliers_boxplot.png")
print("- pertemuan02_scaling_comparison.png")
print("- pertemuan02_eda_analysis.png")
print("- pertemuan02_correlation_heatmap.png")
print("- titanic_cleaned.csv")
