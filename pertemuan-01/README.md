# Pertemuan 1: Pengenalan Python untuk ML dan Setup Environment

## 🎯 Tujuan Pembelajaran

Setelah menyelesaikan pertemuan ini, mahasiswa diharapkan mampu:
1. Menginstall dan mengkonfigurasi environment Python untuk Machine Learning
2. Memahami library dasar yang digunakan dalam ML (NumPy, Pandas, Matplotlib)
3. Menggunakan Jupyter Notebook untuk eksperimen ML
4. Memahami struktur data dasar dalam Python untuk ML

## 📚 Teori Singkat

### Mengapa Python untuk Machine Learning?

Python menjadi bahasa pemrograman pilihan untuk Machine Learning karena:
- **Ecosystem yang kaya**: Library seperti scikit-learn, TensorFlow, PyTorch
- **Syntax yang mudah**: Readable dan mudah dipelajari
- **Community yang besar**: Banyak tutorial, dokumentasi, dan support
- **Integrasi yang baik**: Mudah diintegrasikan dengan tools lain

### Library Fundamental untuk ML

1. **NumPy**: Komputasi numerik dan operasi array
2. **Pandas**: Manipulasi dan analisis data
3. **Matplotlib/Seaborn**: Visualisasi data
4. **Scikit-learn**: Algoritma machine learning

## 🛠️ Setup Environment

### Instalasi Anaconda

1. Download Anaconda dari [https://www.anaconda.com/download](https://www.anaconda.com/download)
2. Install sesuai sistem operasi Anda
3. Verifikasi instalasi:
```bash
conda --version
python --version
```

### Membuat Virtual Environment

```bash
# Membuat environment baru
conda create -n ml-practicum python=3.10

# Aktivasi environment
conda activate ml-practicum

# Install library yang dibutuhkan
conda install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Alternatif: Menggunakan pip

```bash
# Membuat virtual environment
python -m venv ml-env

# Aktivasi (Windows)
ml-env\Scripts\activate

# Aktivasi (Linux/Mac)
source ml-env/bin/activate

# Install library
pip install numpy pandas matplotlib seaborn scikit-learn jupyter notebook
```

## 📝 Praktikum

### Langkah 1: Menjalankan Jupyter Notebook

```bash
jupyter notebook
```

Browser akan otomatis terbuka. Buat notebook baru dengan nama `pertemuan_01_intro.ipynb`

### Langkah 2: Import dan Test Library

```python
# Import library dasar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

# Verifikasi versi
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
```

### Langkah 3: Eksplorasi NumPy

```python
# Membuat array
arr = np.array([1, 2, 3, 4, 5])
print("Array:", arr)
print("Shape:", arr.shape)
print("Data type:", arr.dtype)

# Operasi matematika
print("Mean:", np.mean(arr))
print("Standard deviation:", np.std(arr))

# Array 2D (Matrix)
matrix = np.array([[1, 2, 3], 
                   [4, 5, 6], 
                   [7, 8, 9]])
print("\nMatrix:")
print(matrix)
print("Shape:", matrix.shape)

# Indexing dan Slicing
print("Element at [0,0]:", matrix[0, 0])
print("First row:", matrix[0, :])
print("First column:", matrix[:, 0])
```

### Langkah 4: Eksplorasi Pandas

```python
# Membuat DataFrame
data = {
    'Nama': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Umur': [25, 30, 35, 28, 32],
    'Kota': ['Jakarta', 'Bandung', 'Surabaya', 'Medan', 'Makassar'],
    'Gaji': [5000000, 6000000, 7000000, 5500000, 6500000]
}

df = pd.DataFrame(data)
print(df)

# Operasi dasar DataFrame
print("\n=== Info DataFrame ===")
print(df.info())

print("\n=== Statistik Deskriptif ===")
print(df.describe())

print("\n=== Rata-rata Gaji ===")
print(f"Rp {df['Gaji'].mean():,.0f}")

# Filtering data
print("\n=== Umur > 30 ===")
print(df[df['Umur'] > 30])
```

### Langkah 5: Visualisasi dengan Matplotlib

```python
# Data untuk visualisasi
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Membuat plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)', color='blue', linewidth=2)
plt.title('Fungsi Sinus', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('sin(x)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Bar plot untuk data gaji
plt.figure(figsize=(10, 6))
plt.bar(df['Nama'], df['Gaji'], color='skyblue', edgecolor='navy')
plt.title('Gaji Karyawan', fontsize=14)
plt.xlabel('Nama', fontsize=12)
plt.ylabel('Gaji (Rp)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### Langkah 6: Load Dataset dari Scikit-learn

```python
# Load dataset Iris (dataset klasik untuk ML)
from sklearn.datasets import load_iris

iris = load_iris()

# Convert ke DataFrame
iris_df = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names
)
iris_df['species'] = iris.target
iris_df['species_name'] = iris_df['species'].map({
    0: 'setosa',
    1: 'versicolor', 
    2: 'virginica'
})

print(iris_df.head(10))
print(f"\nJumlah data: {len(iris_df)}")
print(f"Jumlah fitur: {len(iris.feature_names)}")
print(f"Jumlah kelas: {len(iris.target_names)}")

# Visualisasi distribusi
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
iris_df['species_name'].value_counts().plot(kind='bar', color='coral')
plt.title('Distribusi Spesies Iris')
plt.xlabel('Spesies')
plt.ylabel('Jumlah')

plt.subplot(1, 2, 2)
plt.scatter(iris_df['sepal length (cm)'], 
           iris_df['sepal width (cm)'],
           c=iris_df['species'],
           cmap='viridis',
           alpha=0.6)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Scatter Plot Iris Dataset')
plt.colorbar(label='Species')

plt.tight_layout()
plt.show()
```

## 💪 Tugas Praktikum

### Tugas 1: Eksplorasi Array dan Matrix (20 poin)

Buat sebuah notebook yang melakukan:
1. Membuat array NumPy dengan 20 angka random antara 0-100
2. Hitung mean, median, standard deviation, min, dan max
3. Buat matrix 5x5 dengan angka random
4. Hitung transpose dari matrix tersebut
5. Hitung determinant dari matrix (gunakan `np.linalg.det()`)

### Tugas 2: Analisis Data dengan Pandas (30 poin)

Buat DataFrame dengan data minimal 10 mahasiswa yang berisi:
- NIM
- Nama
- Nilai Matematika
- Nilai Fisika
- Nilai Kimia

Lakukan:
1. Hitung rata-rata nilai per mata kuliah
2. Hitung rata-rata nilai per mahasiswa
3. Tampilkan 5 mahasiswa dengan nilai tertinggi
4. Filter mahasiswa dengan rata-rata nilai >= 80
5. Export DataFrame ke file CSV

### Tugas 3: Visualisasi Data (30 poin)

Gunakan data yang sama dari Tugas 2, buat:
1. Bar chart perbandingan nilai rata-rata per mata kuliah
2. Histogram distribusi nilai Matematika
3. Scatter plot hubungan antara Nilai Matematika dan Fisika
4. Box plot untuk semua nilai mata kuliah

### Tugas 4: Eksplorasi Dataset (20 poin)

Load dataset berikut dari scikit-learn:
- `load_wine()` atau `load_breast_cancer()`

Lakukan:
1. Tampilkan 10 baris pertama dalam bentuk DataFrame
2. Tampilkan statistik deskriptif
3. Buat minimal 2 visualisasi yang menunjukkan karakteristik dataset
4. Tulis interpretasi singkat (3-5 kalimat) tentang dataset tersebut

## 📤 Cara Mengumpulkan

1. Buat satu notebook Jupyter (.ipynb) yang berisi semua tugas
2. Pastikan semua cell sudah di-run dan menampilkan output
3. Export notebook ke format PDF (File → Download as → PDF)
4. Rename file: `NIM_Nama_Pertemuan01.pdf`
5. Upload ke sistem LMS atau kirim ke email dosen

**Alternatif**: Push ke GitHub repository pribadi dan share link-nya

## ✅ Kriteria Penilaian

| Aspek | Bobot |
|-------|-------|
| Environment setup berhasil (screenshot) | 10% |
| Tugas 1: Array dan Matrix | 20% |
| Tugas 2: Analisis dengan Pandas | 30% |
| Tugas 3: Visualisasi | 30% |
| Tugas 4: Eksplorasi Dataset | 20% |
| Kerapihan kode dan dokumentasi | 10% |

## 📚 Referensi

1. [NumPy Documentation](https://numpy.org/doc/)
2. [Pandas Documentation](https://pandas.pydata.org/docs/)
3. [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)
4. [Scikit-learn Documentation](https://scikit-learn.org/stable/)
5. [Jupyter Notebook Documentation](https://jupyter-notebook.readthedocs.io/)

## 🆘 Troubleshooting

### Error: "conda: command not found"
- Pastikan Anaconda sudah terinstall dengan benar
- Restart terminal/command prompt
- Cek PATH environment variable

### Error saat import library
```bash
# Install ulang library
conda install -c conda-forge <nama-library>
# atau
pip install <nama-library>
```

### Jupyter Notebook tidak bisa dibuka
```bash
# Update Jupyter
pip install --upgrade jupyter notebook
```

---

**Selamat Belajar! 🚀**

Jika ada pertanyaan, silakan diskusikan di forum kelas atau hubungi asisten praktikum.
