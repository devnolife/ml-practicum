# Pertemuan 2: Data Preprocessing dan Exploratory Data Analysis (EDA)

## 🎯 Tujuan Pembelajaran

Setelah menyelesaikan pertemuan ini, mahasiswa diharapkan mampu:
1. Memahami pentingnya preprocessing dalam machine learning
2. Melakukan data cleaning (handling missing values, duplicates, outliers)
3. Melakukan normalisasi dan standardisasi data
4. Melakukan Exploratory Data Analysis (EDA) untuk memahami karakteristik data
5. Membuat visualisasi yang informatif untuk analisis data

## 📚 Teori Singkat

### Mengapa Data Preprocessing Penting?

> "Garbage In, Garbage Out" - Kualitas model ML sangat bergantung pada kualitas data

Data di dunia nyata sering kali:
- **Tidak lengkap**: Missing values
- **Tidak konsisten**: Format berbeda, duplikasi
- **Noisy**: Outliers, error pengukuran
- **Tidak seimbang**: Skala fitur yang berbeda jauh

### Tahapan Data Preprocessing

1. **Data Cleaning**: Mengatasi missing values, duplikasi, outliers
2. **Data Transformation**: Normalisasi, standardisasi, encoding
3. **Feature Engineering**: Membuat fitur baru dari fitur yang ada
4. **Data Splitting**: Membagi data untuk training dan testing

### Exploratory Data Analysis (EDA)

EDA adalah proses investigasi awal untuk:
- Memahami struktur data
- Menemukan pola dan anomali
- Memvalidasi asumsi
- Menentukan fitur yang relevan

## 📝 Praktikum

### Persiapan: Import Library

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# Setting style untuk visualisasi
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
```

### Langkah 1: Load dan Inspeksi Data

```python
# Load dataset (contoh: menggunakan dataset Titanic dari seaborn)
df = sns.load_dataset('titanic')

# Inspeksi awal
print("=== 5 Baris Pertama ===")
print(df.head())

print("\n=== Info Dataset ===")
print(df.info())

print("\n=== Statistik Deskriptif ===")
print(df.describe())

print("\n=== Ukuran Dataset ===")
print(f"Jumlah baris: {df.shape[0]}")
print(f"Jumlah kolom: {df.shape[1]}")

print("\n=== Nama Kolom ===")
print(df.columns.tolist())
```

### Langkah 2: Handling Missing Values

```python
# Cek missing values
print("=== Missing Values ===")
missing = df.isnull().sum()
print(missing[missing > 0])

# Visualisasi missing values
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# Strategi 1: Drop baris dengan missing values
df_dropped = df.dropna()
print(f"\nJumlah baris setelah drop: {len(df_dropped)}")

# Strategi 2: Fill dengan nilai tertentu
df_filled = df.copy()
df_filled['age'].fillna(df_filled['age'].median(), inplace=True)
df_filled['embarked'].fillna(df_filled['embarked'].mode()[0], inplace=True)

# Strategi 3: Menggunakan SimpleImputer dari sklearn
imputer = SimpleImputer(strategy='mean')
df_numeric = df.select_dtypes(include=[np.number])
df_imputed = pd.DataFrame(
    imputer.fit_transform(df_numeric),
    columns=df_numeric.columns
)

print("\n=== Missing Values Setelah Handling ===")
print(df_filled.isnull().sum().sum())
```

### Langkah 3: Handling Duplicates

```python
# Cek duplikasi
duplicates = df.duplicated().sum()
print(f"Jumlah baris duplikat: {duplicates}")

# Tampilkan baris duplikat
if duplicates > 0:
    print("\n=== Baris Duplikat ===")
    print(df[df.duplicated(keep=False)])
    
# Remove duplikasi
df_clean = df.drop_duplicates()
print(f"\nJumlah baris setelah remove duplikat: {len(df_clean)}")
```

### Langkah 4: Handling Outliers

```python
# Deteksi outliers menggunakan IQR (Interquartile Range)
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Contoh: deteksi outliers pada kolom 'fare'
df_temp = df_filled.copy()
outliers, lower, upper = detect_outliers_iqr(df_temp, 'fare')
print(f"\n=== Outliers di kolom 'fare' ===")
print(f"Jumlah outliers: {len(outliers)}")
print(f"Lower bound: {lower:.2f}")
print(f"Upper bound: {upper:.2f}")

# Visualisasi outliers dengan boxplot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sebelum handling outliers
axes[0].boxplot(df_temp['fare'].dropna())
axes[0].set_title('Fare - Sebelum Handling Outliers')
axes[0].set_ylabel('Fare')

# Setelah handling (capping method)
df_temp['fare_capped'] = df_temp['fare'].clip(lower=lower, upper=upper)
axes[1].boxplot(df_temp['fare_capped'])
axes[1].set_title('Fare - Setelah Handling Outliers')
axes[1].set_ylabel('Fare')

plt.tight_layout()
plt.show()
```

### Langkah 5: Normalisasi dan Standardisasi

```python
# Pilih kolom numerik
numeric_cols = ['age', 'fare']
df_numeric = df_filled[numeric_cols].dropna()

# Min-Max Normalization (0-1)
scaler_minmax = MinMaxScaler()
df_normalized = pd.DataFrame(
    scaler_minmax.fit_transform(df_numeric),
    columns=numeric_cols
)

# Standardization (mean=0, std=1)
scaler_standard = StandardScaler()
df_standardized = pd.DataFrame(
    scaler_standard.fit_transform(df_numeric),
    columns=numeric_cols
)

# Visualisasi perbandingan
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Original
axes[0].hist(df_numeric['age'], bins=30, edgecolor='black', alpha=0.7)
axes[0].set_title('Original Data - Age')
axes[0].set_xlabel('Age')

# Normalized
axes[1].hist(df_normalized['age'], bins=30, edgecolor='black', alpha=0.7, color='green')
axes[1].set_title('Normalized Data - Age')
axes[1].set_xlabel('Age (normalized)')

# Standardized
axes[2].hist(df_standardized['age'], bins=30, edgecolor='black', alpha=0.7, color='orange')
axes[2].set_title('Standardized Data - Age')
axes[2].set_xlabel('Age (standardized)')

plt.tight_layout()
plt.show()

print("=== Statistik Comparison ===")
print(f"Original - Mean: {df_numeric['age'].mean():.2f}, Std: {df_numeric['age'].std():.2f}")
print(f"Normalized - Mean: {df_normalized['age'].mean():.2f}, Std: {df_normalized['age'].std():.2f}")
print(f"Standardized - Mean: {df_standardized['age'].mean():.2f}, Std: {df_standardized['age'].std():.2f}")
```

### Langkah 6: Exploratory Data Analysis (EDA)

```python
# 1. Distribusi Univariate
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram Age
axes[0, 0].hist(df_filled['age'], bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Distribusi Umur Penumpang')
axes[0, 0].set_xlabel('Umur')
axes[0, 0].set_ylabel('Frekuensi')

# Bar plot Survival
survival_counts = df_filled['survived'].value_counts()
axes[0, 1].bar(survival_counts.index, survival_counts.values, color=['red', 'green'])
axes[0, 1].set_title('Jumlah Survivor vs Non-Survivor')
axes[0, 1].set_xlabel('Survived (0=No, 1=Yes)')
axes[0, 1].set_ylabel('Count')

# Bar plot Class
class_counts = df_filled['class'].value_counts()
axes[1, 0].bar(class_counts.index, class_counts.values, color='skyblue')
axes[1, 0].set_title('Distribusi Passenger Class')
axes[1, 0].set_xlabel('Class')
axes[1, 0].set_ylabel('Count')

# Pie chart Gender
gender_counts = df_filled['sex'].value_counts()
axes[1, 1].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
axes[1, 1].set_title('Distribusi Gender')

plt.tight_layout()
plt.show()

# 2. Analisis Bivariate
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Survival rate by Gender
survival_gender = pd.crosstab(df_filled['sex'], df_filled['survived'], normalize='index') * 100
survival_gender.plot(kind='bar', ax=axes[0], color=['red', 'green'])
axes[0].set_title('Survival Rate by Gender')
axes[0].set_xlabel('Gender')
axes[0].set_ylabel('Percentage (%)')
axes[0].legend(['Not Survived', 'Survived'])
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

# Survival rate by Class
survival_class = pd.crosstab(df_filled['class'], df_filled['survived'], normalize='index') * 100
survival_class.plot(kind='bar', ax=axes[1], color=['red', 'green'])
axes[1].set_title('Survival Rate by Class')
axes[1].set_xlabel('Class')
axes[1].set_ylabel('Percentage (%)')
axes[1].legend(['Not Survived', 'Survived'])
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

# Age distribution by Survival
axes[2].hist([df_filled[df_filled['survived']==0]['age'].dropna(), 
              df_filled[df_filled['survived']==1]['age'].dropna()], 
             bins=30, label=['Not Survived', 'Survived'], color=['red', 'green'], alpha=0.7)
axes[2].set_title('Age Distribution by Survival')
axes[2].set_xlabel('Age')
axes[2].set_ylabel('Frequency')
axes[2].legend()

plt.tight_layout()
plt.show()

# 3. Correlation Matrix
numeric_features = df_filled.select_dtypes(include=[np.number])
correlation_matrix = numeric_features.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()
```

## 💪 Tugas Praktikum

### Tugas 1: Data Cleaning (30 poin)

Download dataset dari [Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) atau gunakan dataset lain yang memiliki missing values.

Lakukan:
1. Load dataset dan tampilkan info dasar (shape, columns, dtypes)
2. Identifikasi dan visualisasi missing values
3. Handling missing values dengan 3 strategi berbeda
4. Identifikasi dan remove duplicates (jika ada)
5. Dokumentasikan keputusan Anda dalam markdown cell

### Tugas 2: Outlier Detection dan Handling (25 poin)

Gunakan dataset yang sama:
1. Pilih minimal 3 kolom numerik
2. Deteksi outliers menggunakan metode IQR
3. Visualisasi outliers dengan boxplot (before & after)
4. Implementasikan 2 metode handling outliers:
   - Capping/Clipping
   - Removal
5. Bandingkan hasil kedua metode dengan visualisasi

### Tugas 3: Normalisasi dan Standardisasi (20 poin)

1. Pilih 5 kolom numerik dari dataset
2. Terapkan Min-Max Normalization
3. Terapkan Standardization
4. Buat visualisasi perbandingan (histogram atau distribution plot)
5. Jelaskan kapan menggunakan normalisasi vs standardisasi

### Tugas 4: Exploratory Data Analysis (25 poin)

Lakukan EDA komprehensif:
1. **Univariate Analysis**: Buat minimal 4 visualisasi untuk fitur individual
2. **Bivariate Analysis**: Analisis hubungan antara 2 variabel (minimal 3 pasang)
3. **Multivariate Analysis**: Buat correlation matrix dan heatmap
4. **Insights**: Tulis minimal 5 insight/temuan menarik dari data
5. **Recommendations**: Berikan rekomendasi fitur mana yang penting untuk modeling

## 📤 Cara Mengumpulkan

1. Buat notebook Jupyter (.ipynb) dengan struktur yang jelas
2. Setiap section harus ada penjelasan dalam markdown cell
3. Pastikan semua visualisasi ter-render dengan baik
4. Export ke PDF: `NIM_Nama_Pertemuan02.pdf`
5. Upload ke LMS atau push ke GitHub repository

## ✅ Kriteria Penilaian

| Aspek | Bobot |
|-------|-------|
| Tugas 1: Data Cleaning | 30% |
| Tugas 2: Outlier Handling | 25% |
| Tugas 3: Scaling | 20% |
| Tugas 4: EDA & Insights | 25% |
| Kualitas visualisasi | 15% |
| Dokumentasi & penjelasan | 15% |
| Kerapihan kode | 10% |

## 📚 Referensi

1. [Pandas User Guide - Missing Data](https://pandas.pydata.org/docs/user_guide/missing_data.html)
2. [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
3. [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
4. [Outlier Detection Methods](https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba)

## 💡 Tips

1. **Missing Values**: Jangan langsung drop! Pertimbangkan apakah missing itu random atau ada pola
2. **Outliers**: Tidak semua outliers perlu dihilangkan. Kadang outliers adalah data penting
3. **Scaling**: Normalisasi untuk algoritma distance-based (KNN), standardisasi untuk algoritma gradient-based (Linear Regression)
4. **EDA**: Visualisasi adalah kunci untuk memahami data. Jangan skip tahap ini!

---

**Happy Data Wrangling! 🧹📊**
