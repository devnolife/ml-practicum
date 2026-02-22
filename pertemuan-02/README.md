# Pertemuan 2: Membersihkan Data dan Eksplorasi Data (EDA)

## 🎯 Tujuan Pembelajaran

Setelah pertemuan ini, kamu bisa:
1. Membersihkan data yang kotor (missing values, duplikat)
2. Mendeteksi dan menangani outliers (data aneh)
3. Menskala data agar siap dipakai ML
4. Membuat visualisasi untuk memahami data

## 📚 Penjelasan Singkat

### Kenapa Harus Bersihkan Data?

Data di dunia nyata biasanya kotor:
- **Data hilang**: Ada nilai yang kosong
- **Data duplikat**: Data yang sama muncul 2 kali
- **Outliers**: Nilai yang aneh/ekstrim
- **Skala berbeda**: Umur (1-100) vs Gaji (1000000-10000000)

**Ingat**: Model ML hanya sebagus data yang kita kasih!

## 📝 Praktikum

### Persiapan

> **💡 Penjelasan Program:**
> - **Tujuan**: Import library untuk data cleaning, preprocessing, dan visualisasi
> - **Kapan digunakan**: Di awal setiap project ML sebelum proses data cleaning
> - **Penjelasan Library**:
>   - `StandardScaler`: Standarisasi data (mean=0, std=1) untuk algoritma sensitive terhadap scale
>   - `MinMaxScaler`: Normalisasi data ke range 0-1
>   - Library lain: NumPy, Pandas untuk manipulasi data; Matplotlib, Seaborn untuk visualisasi

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
```

### Langkah 1: Load Data dan Lihat Sekilas

> **💡 Penjelasan Program:**
> - **Tujuan**: Load dataset dan lakukan inspeksi awal untuk memahami struktur data
> - **Kapan digunakan**: Langkah pertama dalam setiap project data science untuk familiarisasi dengan data
> - **Penjelasan Kode**:
>   - `sns.load_dataset('titanic')`: Load dataset Titanic yang sudah ada di Seaborn
>   - `.head()`: Tampilkan 5 baris pertama untuk preview data
>   - `len(df)`: Hitung jumlah baris (sampel/observasi)
>   - `len(df.columns)`: Hitung jumlah kolom (fitur/variabel)
> - **Dataset Titanic**: Data penumpang kapal Titanic (selamat/tidak, umur, kelas, dll)

```python
# Load dataset Titanic
df = sns.load_dataset('titanic')

# Lihat 5 baris pertama
print(df.head())

# Info dataset
print(f"\nJumlah baris: {len(df)}")
print(f"Jumlah kolom: {len(df.columns)}")
```

### Langkah 2: Cari Data yang Hilang

> **💡 Penjelasan Program:**
> - **Tujuan**: Identifikasi kolom yang memiliki missing values (data kosong/null)
> - **Kapan digunakan**: Setelah load data, sebelum training model (model tidak bisa proses data null)
> - **Penjelasan Kode**:
>   - `df.isnull()`: Return DataFrame boolean (True jika null, False jika tidak)
>   - `.sum()`: Hitung jumlah True per kolom (jumlah missing values)
>   - `sns.heatmap()`: Visualisasi missing values dengan peta warna (kuning=hilang, ungu=ada)
>   - `yticklabels=False`: Sembunyikan label baris untuk readability
> - **Kenapa penting**: Missing values bisa bikin model error atau hasil prediksi buruk

```python
# Cek data kosong
print("Data yang hilang:")
print(df.isnull().sum())

# Visualisasi dengan heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False)
plt.title('Peta Data Hilang (Kuning = Hilang)')
plt.show()
```

### Langkah 3: Isi Data yang Hilang

> **💡 Penjelasan Program:**
> - **Tujuan**: Menangani missing values dengan strategi yang tepat agar data siap untuk ML
> - **Kapan digunakan**: Setelah identifikasi missing values, sebelum training
> - **Strategi Imputation**:
>   - **Median** (`fillna(median())`): Untuk data numerik dengan outliers (robust)
>   - **Mode** (`fillna(mode()[0])`): Untuk data kategorikal (nilai paling sering muncul)
>   - **Drop** (`dropna()`): Buang baris dengan missing values (jika sedikit)
> - **Penjelasan Parameter**:
>   - `inplace=True`: Ubah DataFrame original (tidak buat copy baru)
> - **Tips**: Jangan drop jika missing values > 5% data!

```python
# Cara 1: Isi dengan nilai tengah (median)
df['age'].fillna(df['age'].median(), inplace=True)

# Cara 2: Isi dengan nilai paling sering (mode)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Cara 3: Hapus baris yang masih kosong
df = df.dropna()

print("Data hilang setelah dibersihkan:", df.isnull().sum().sum())
```

### Langkah 4: Hapus Data Duplikat

> **💡 Penjelasan Program:**
> - **Tujuan**: Menghilangkan baris yang identik (duplikat) untuk menghindari bias dalam model
> - **Kapan digunakan**: Setelah handle missing values, saat ada kemungkinan data entry ganda
> - **Penjelasan Kode**:
>   - `df.duplicated()`: Return Series boolean (True jika baris duplikat)
>   - `.sum()`: Hitung total baris duplikat
>   - `drop_duplicates()`: Hapus semua baris duplikat, keep yang pertama
> - **Kenapa penting**: Data duplikat bisa bikin model "belajar terlalu banyak" dari satu observasi yang sama
> - **Note**: By default, semua kolom dipertimbangkan untuk cek duplikat

```python
# Cek dan hapus duplikat
print(f"Jumlah duplikat: {df.duplicated().sum()}")
df = df.drop_duplicates()
print(f"Jumlah data sekarang: {len(df)}")
```

### Langkah 5: Deteksi dan Tangani Outliers

> **💡 Penjelasan Program:**
> - **Tujuan**: Identifikasi data yang ekstrim/aneh (outliers) yang bisa ganggu performa model
> - **Kapan digunakan**: Untuk data numerik continuous, setelah data cleaning dasar
> - **Penjelasan Kode**:
>   - `plt.boxplot()`: Visualisasi distribusi data dan outliers
>   - Boxplot menunjukkan: Q1 (25%), Median (50%), Q3 (75%), dan outliers (titik di luar whiskers)
>   - `plt.subplot(1, 2, x)`: Buat 2 grafik side-by-side (1 baris, 2 kolom)
> - **Apa itu Outlier**: Data yang jauh dari mayoritas data (contoh: umur 200 tahun, harga tiket $1 juta)
> - **Strategi**: Bisa di-cap (batasi), remove, atau transform tergantung kasus

```python
# Lihat outliers pakai boxplot
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.boxplot(df['age'].dropna())
plt.title('Boxplot: Umur')

plt.subplot(1, 2, 2)
plt.boxplot(df['fare'].dropna())
plt.title('Boxplot: Harga Tiket')

plt.tight_layout()
plt.show()

# Batasi outliers (capping)
Q1 = df['fare'].quantile(0.25)
Q3 = df['fare'].quantile(0.75)
IQR = Q3 - Q1
upper_limit = Q3 + 1.5 * IQR

df['fare'] = df['fare'].clip(upper=upper_limit)
print(f"Harga maksimal setelah capping: {df['fare'].max()}")
```

### Langkah 6: Skala Data (Normalisasi)

> **💡 Penjelasan Program:**
> - **Tujuan**: Ubah skala data ke range yang sama (0-1) agar semua fitur punya kontribusi seimbang
> - **Kapan digunakan**: Sebelum training algoritma yang sensitive terhadap scale (KNN, SVM, Neural Networks)
> - **Kenapa penting**: Tanpa normalisasi, fitur dengan nilai besar (misal: gaji 5000000) akan dominan vs fitur kecil (umur 25)
> - **Penjelasan Kode**:
>   - `MinMaxScaler()`: Transformasi data ke range [0, 1] dengan rumus: (x - min) / (max - min)
>   - `.fit_transform()`: Hitung min/max dari data, lalu transform
>   - `.copy()`: Buat copy DataFrame agar original tidak berubah
> - **Alternative**: StandardScaler (mean=0, std=1) untuk data dengan distribusi normal

```python
# Pilih kolom angka
data_angka = df[['age', 'fare']].copy()

# Normalisasi ke skala 0-1
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data_angka)
df_norm = pd.DataFrame(data_normalized, columns=['age', 'fare'])

# Bandingkan
print("SEBELUM Normalisasi:")
print(data_angka.head())
print("\nSESUDAH Normalisasi:")
print(df_norm.head())
```

### Langkah 7: Eksplorasi Data (EDA)

```python
# 1. Lihat distribusi data
plt.figure(figsize=(12, 8))

# Umur penumpang
plt.subplot(2, 2, 1)
plt.hist(df['age'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribusi Umur')

# Selamat atau tidak
plt.subplot(2, 2, 2)
df['survived'].value_counts().plot(kind='bar', color=['red', 'green'])
plt.title('Selamat vs Tidak')
plt.xticks(rotation=0)

# Kelas penumpang
plt.subplot(2, 2, 3)
df['class'].value_counts().plot(kind='bar', color='orange')
plt.title('Kelas Penumpang')
plt.xticks(rotation=0)

# Gender
plt.subplot(2, 2, 4)
df['sex'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Gender')

plt.tight_layout()
plt.show()

# 2. Analisis hubungan antar variabel
plt.figure(figsize=(12, 5))

# Tingkat selamat berdasarkan gender
plt.subplot(1, 2, 1)
survival_gender = pd.crosstab(df['sex'], df['survived'], normalize='index') * 100
survival_gender.plot(kind='bar', color=['red', 'green'])
plt.title('Tingkat Selamat per Gender (%)')
plt.legend(['Tidak', 'Selamat'])
plt.xticks(rotation=0)

# Tingkat selamat berdasarkan kelas
plt.subplot(1, 2, 2)
survival_class = pd.crosstab(df['class'], df['survived'], normalize='index') * 100
survival_class.plot(kind='bar', color=['red', 'green'])
plt.title('Tingkat Selamat per Kelas (%)')
plt.legend(['Tidak', 'Selamat'])
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

# 3. Correlation Matrix
kolom_angka = df.select_dtypes(include=[np.number])
korelasi = kolom_angka.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(korelasi, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Korelasi Antar Variabel')
plt.show()
```

## 💪 Tugas Praktikum

### Tugas 1: Bersihkan Data (40 poin)

Gunakan dataset 'tips' dari seaborn:
```python
df = sns.load_dataset('tips')
# Atau download dataset lain dari Kaggle
```

Lakukan:
1. Tampilkan jumlah data hilang
2. Isi atau hapus data hilang
3. Cek dan hapus duplikat
4. Tunjukkan hasil akhir

### Tugas 2: Tangani Outliers (30 poin)

```python
# 1. Buat boxplot untuk 2 kolom numerik
# 2. Tangani outliers dengan capping
# 3. Buat boxplot before & after
```

### Tugas 3: Eksplorasi Data (30 poin)

Buat minimal 4 visualisasi:
1. Histogram untuk 2 kolom numerik
2. Bar chart untuk 1 kolom kategori
3. Crosstab untuk analisis hubungan

Tulis 3 insight yang kamu temukan!

## ✅ Cara Mengumpulkan

Simpan dalam notebook (.ipynb), export ke PDF: `NIM_Nama_Pertemuan02.pdf`

## 🔑 Cheat Sheet

```python
# Cek data hilang
df.isnull().sum()

# Isi data hilang
df['kolom'].fillna(df['kolom'].median(), inplace=True)

# Hapus duplikat
df = df.drop_duplicates()

# Normalisasi
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[['kolom']])

# Boxplot
plt.boxplot(df['kolom'])
```

---

**Selamat Membersihkan Data! 🧹✨**
