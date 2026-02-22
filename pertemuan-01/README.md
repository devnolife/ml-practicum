# Pertemuan 1: Pengenalan Python untuk Machine Learning

## 🎯 Tujuan Pembelajaran

Setelah pertemuan ini, kamu bisa:
1. Install Python dan library ML
2. Pakai NumPy, Pandas, dan Matplotlib
3. Membuat visualisasi data sederhana

## 📚 Pengenalan Singkat

### Kenapa Pakai Python?

Python itu mudah dipelajari dan punya banyak library untuk Machine Learning:
- **NumPy**: Untuk menghitung angka dan array
- **Pandas**: Untuk mengolah data (seperti Excel)
- **Matplotlib**: Untuk membuat grafik
- **Scikit-learn**: Untuk algoritma ML

## 🛠️ Setup - Install Library

### Cara 1: Menggunakan pip (Lebih Sederhana)

> **💡 Penjelasan Program:**
> - **Tujuan**: Install semua library Python yang dibutuhkan untuk Machine Learning dalam satu perintah
> - **Kapan digunakan**: Saat pertama kali setup environment Python untuk ML, atau saat library belum terinstall
> - **Library yang diinstall**:
>   - `numpy`: Untuk operasi matematika dan array
>   - `pandas`: Untuk manipulasi data tabular (seperti Excel)
>   - `matplotlib`: Untuk membuat visualisasi dasar
>   - `seaborn`: Untuk visualisasi statistik yang lebih cantik
>   - `scikit-learn`: Kumpulan algoritma Machine Learning
>   - `jupyter`: Untuk menjalankan notebook interaktif

```bash
# Install semua library sekaligus
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Cara 2: Menggunakan Anaconda (Opsional)

> **💡 Penjelasan Program:**
> - **Tujuan**: Membuat environment Python yang terisolasi khusus untuk project ML
> - **Kapan digunakan**: Saat ingin memisahkan dependencies project berbeda, atau menggunakan distribusi Anaconda
> - **Penjelasan Kode**:
>   - `conda create -n ml-practicum python=3.10`: Buat environment baru bernama "ml-practicum" dengan Python 3.10
>   - `conda activate ml-practicum`: Aktifkan environment tersebut
>   - `conda install ...`: Install library yang dibutuhkan di dalam environment

```bash
conda create -n ml-practicum python=3.10
conda activate ml-practicum
conda install numpy pandas matplotlib seaborn scikit-learn jupyter
```

## 📝 Praktikum

### Langkah 1: Jalankan Jupyter Notebook

```bash
jupyter notebook
```

Browser akan terbuka otomatis. Buat notebook baru.

### Langkah 2: Import Library

> **💡 Penjelasan Program:**
> - **Tujuan**: Mengimport library yang sudah diinstall agar bisa digunakan dalam kode Python
> - **Kapan digunakan**: Di awal setiap script atau notebook Python untuk ML
> - **Penjelasan Kode**:
>   - `import numpy as np`: Import NumPy dengan alias 'np' (konvensi standar)
>   - `import pandas as pd`: Import Pandas dengan alias 'pd' (konvensi standar)
>   - `import matplotlib.pyplot as plt`: Import modul pyplot dari Matplotlib untuk membuat grafik
>   - Jika tidak ada error, berarti library berhasil diinstall dengan benar

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Library berhasil di-import!")
```

### Langkah 3: Belajar NumPy (Untuk Hitung Angka)

> **💡 Penjelasan Program:**
> - **Tujuan**: Melakukan operasi matematika cepat pada array dan matrix
> - **Kapan digunakan**: Saat perlu perhitungan numerik, operasi matrix, atau array multi-dimensi dalam ML
> - **Penjelasan Kode**:
>   - `np.array([...])`: Membuat array NumPy dari list Python (lebih cepat dari list biasa)
>   - `.mean()`: Menghitung rata-rata (average) dari semua elemen
>   - `.sum()`: Menjumlahkan semua elemen array
>   - Array 2D (matrix): Digunakan untuk representasi data ML (baris = sampel, kolom = fitur)
> - **Keuntungan**: 10-100x lebih cepat dari list Python untuk operasi matematika

```python
# Buat array sederhana
angka = np.array([10, 20, 30, 40, 50])
print("Array:", angka)

# Hitung rata-rata dan total
print("Rata-rata:", angka.mean())
print("Total:", angka.sum())

# Buat matrix 3x3
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
print("\nMatrix:")
print(matrix)
```

### Langkah 4: Belajar Pandas (Untuk Kelola Data)

> **💡 Penjelasan Program:**
> - **Tujuan**: Mengolah data tabular (seperti Excel) dengan mudah - membaca, filter, transform data
> - **Kapan digunakan**: Untuk exploratory data analysis, data cleaning, preprocessing sebelum ML
> - **Penjelasan Kode**:
>   - `pd.DataFrame(dict)`: Membuat tabel data dari dictionary (dict keys = nama kolom)
>   - `df['Nilai'].mean()`: Akses kolom 'Nilai' dan hitung rata-ratanya
>   - `df[df['Umur'] > 21]`: Filter baris yang memenuhi kondisi (boolean indexing)
>   - DataFrame adalah struktur data utama dalam analisis data Python
> - **Analogi**: Pandas DataFrame ≈ Excel Spreadsheet yang bisa diprogram

```python
# Buat tabel data
data = {
    'Nama': ['Ali', 'Budi', 'Citra', 'Dina'],
    'Umur': [20, 22, 21, 23],
    'Nilai': [85, 90, 88, 92]
}

df = pd.DataFrame(data)
print(df)

# Hitung rata-rata nilai
print("\nRata-rata nilai:", df['Nilai'].mean())

# Filter yang umurnya > 21
print("\nYang umurnya > 21:")
print(df[df['Umur'] > 21])
```

### Langkah 5: Buat Grafik Sederhana

> **💡 Penjelasan Program:**
> - **Tujuan**: Visualisasi data untuk memahami pola, trend, dan distribusi data dengan mudah
> - **Kapan digunakan**: Untuk presentasi hasil, eksplorasi data, atau memahami performa model ML
> - **Penjelasan Kode**:
>   - `plt.figure(figsize=(8, 5))`: Buat canvas kosong ukuran 8x5 inch
>   - `plt.bar(x, y)`: Buat bar chart (grafik batang)
>   - `plt.title()`, `plt.xlabel()`, `plt.ylabel()`: Tambah label untuk kejelasan
>   - `plt.show()`: Tampilkan grafik di layar
> - **Kenapa penting**: "A picture is worth a thousand words" - grafik memudahkan pemahaman data

```python
# Data nilai siswa
nama = ['Ali', 'Budi', 'Citra', 'Dina']
nilai = [85, 90, 88, 92]

# Buat grafik batang
plt.figure(figsize=(8, 5))
plt.bar(nama, nilai, color='skyblue')
plt.title('Nilai Siswa')
plt.xlabel('Nama')
plt.ylabel('Nilai')
plt.show()
```

### Langkah 6: Load Dataset Iris (Dataset Bunga)

> **💡 Penjelasan Program:**
> - **Tujuan**: Memuat dataset bawaan scikit-learn (Iris flower dataset) untuk belajar ML
> - **Kapan digunakan**: Saat belajar algoritma classification atau ingin dataset sederhana untuk testing
> - **Penjelasan Kode**:
>   - `load_iris()`: Load dataset Iris (150 sampel, 4 fitur, 3 spesies bunga)
>   - `pd.DataFrame(iris.data, columns=...)`: Konversi array NumPy ke Pandas DataFrame
>   - `plt.scatter(x, y, c=color)`: Buat scatter plot dengan warna berbeda per kategori
>   - `c=df_iris['jenis']`: Warna titik berdasarkan spesies (0=setosa, 1=versicolor, 2=virginica)
> - **Dataset Iris**: Dataset klasik untuk belajar classification (sangat populer sejak 1936!)

```python
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris['jenis'] = iris.target

print(df_iris.head())

# Buat scatter plot
plt.scatter(df_iris.iloc[:, 0], df_iris.iloc[:, 1], c=df_iris['jenis'])
plt.xlabel('Panjang Sepal')
plt.ylabel('Lebar Sepal')
plt.title('Data Bunga Iris')
plt.show()
```

## 💪 Tugas Praktikum

### Tugas 1: Main dengan NumPy (25 poin)

```python
# Buat array dengan 10 angka random
angka = np.random.randint(1, 100, 10)
print("Data:", angka)
print("Rata-rata:", angka.mean())
print("Nilai tertinggi:", angka.max())
print("Nilai terendah:", angka.min())
```

### Tugas 2: Buat Tabel Data (35 poin)

Buat tabel data 5 teman kamu dengan kolom: Nama, Umur, Nilai
```python
data = {
    'Nama': ['...', '...', ...],
    'Umur': [...],
    'Nilai': [...]
}
df = pd.DataFrame(data)
```

Lalu:
- Tampilkan yang nilainya >= 80
- Hitung rata-rata umur
- Simpan ke file CSV: `df.to_csv('data_siswa.csv')`

### Tugas 3: Buat Grafik (25 poin)

Buat 2 grafik dari data Tugas 2:
1. Grafik batang untuk nilai semua siswa
2. Scatter plot umur vs nilai

### Tugas 4: Eksplorasi Dataset (15 poin)

```python
from sklearn.datasets import load_wine
wine = load_wine()
df_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
```

Tampilkan:
- 5 baris pertama
- Statistik (mean, min, max) dari 3 kolom pertama

## ✅ Cara Mengumpulkan

1. Simpan semua kode dalam 1 file notebook (.ipynb)
2. Export ke PDF
3. Upload dengan nama: `NIM_Nama_Pertemuan01.pdf`

## 📚 Referensi Cepat

- [NumPy Cheatsheet](https://numpy.org/doc/)
- [Pandas Cheatsheet](https://pandas.pydata.org/docs/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)

---

**Selamat Belajar! 🚀**
