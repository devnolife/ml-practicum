# Praktikum Applied Machine Learning

**Kode Mata Kuliah:** CW6552021547  
**Semester:** V (Lima)  
**SKS:** 3 SKS  
**Program Studi:** Informatika  
**Fakultas:** Teknik  
**Universitas:** Universitas Muhammadiyah Makassar

---

## 📘 Deskripsi

Repositori ini berisi materi praktikum Applied Machine Learning yang mencakup prinsip-prinsip inti dan aplikasi dari supervised learning, unsupervised learning, dan reinforcement learning. Mahasiswa akan menggunakan pustaka Python standar industri seperti scikit-learn untuk memecahkan masalah dunia nyata.

## 🎯 Capaian Pembelajaran

Setelah menyelesaikan mata kuliah ini, mahasiswa diharapkan mampu:

1. Memahami dan mengimplementasikan algoritma supervised learning (regresi, klasifikasi)
2. Menguasai teknik unsupervised learning (clustering, dimensionality reduction)
3. Mampu menerapkan reinforcement learning untuk problem-solving
4. Menggunakan scikit-learn dan library Python untuk machine learning
5. Melakukan evaluasi model dan optimasi hyperparameter

## 📚 Struktur Materi

Repositori ini mencakup materi untuk **8 pertemuan pertama** (sampai UTS):

| Pertemuan | Topik | Materi Utama |
|-----------|-------|--------------|
| [01](./pertemuan-01) | **Python untuk ML & Setup Environment** | Install tools, NumPy, Pandas, Matplotlib, Jupyter Notebook |
| [02](./pertemuan-02) | **Data Preprocessing & EDA** | Cleaning, normalisasi, visualisasi, handling missing values & outliers |
| [03](./pertemuan-03) | **Linear & Polynomial Regression** | Simple/Multiple Linear Regression, prediksi harga rumah |
| [04](./pertemuan-04) | **Classification** | Logistic Regression, Decision Tree, klasifikasi dataset Iris |
| [05](./pertemuan-05) | **Support Vector Machine** | SVM dengan kernel tricks, parameter tuning |
| [06](./pertemuan-06) | **Ensemble Methods** | Random Forest, Gradient Boosting, prediksi churn |
| [07](./pertemuan-07) | **Clustering** | K-Means, Hierarchical Clustering, segmentasi pelanggan |
| [08](./pertemuan-08) | **UTS: Mid-semester Project** | Mini project ML end-to-end |

## 🚀 Getting Started

### Prerequisites

Pastikan Anda telah menginstall:
- Python 3.8 atau lebih baru
- Anaconda atau Miniconda (recommended)
- Jupyter Notebook atau JupyterLab
- Git (untuk clone repository)

### Installation

1. **Clone repository:**
```bash
git clone https://github.com/[your-username]/ml-practicum.git
cd ml-practicum
```

2. **Buat virtual environment:**
```bash
# Menggunakan conda
conda create -n ml-practicum python=3.10
conda activate ml-practicum

# Atau menggunakan venv
python -m venv ml-env
source ml-env/bin/activate  # Linux/Mac
ml-env\Scripts\activate     # Windows
```

3. **Install dependencies:**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter notebook
```

4. **Jalankan Jupyter Notebook:**
```bash
jupyter notebook
```

### Alternatif: Google Colab

Jika tidak ingin install di local, Anda bisa menggunakan [Google Colab](https://colab.research.google.com/):
1. Upload notebook ke Google Drive
2. Buka dengan Google Colab
3. Install library yang diperlukan di cell pertama:
```python
!pip install scikit-learn matplotlib seaborn
```

## 📖 Cara Menggunakan Repository

### Untuk Setiap Pertemuan:

1. **Baca README.md** di folder pertemuan untuk memahami:
   - Tujuan pembelajaran
   - Teori singkat
   - Step-by-step praktikum

2. **Ikuti Tutorial** yang ada di README.md:
   - Copy-paste code ke Jupyter Notebook
   - Run dan pahami hasilnya
   - Experiment dengan parameter berbeda

3. **Kerjakan Tugas** yang ada di bagian akhir README:
   - Setiap tugas memiliki bobot nilai
   - Kerjakan di notebook terpisah
   - Submit sesuai instruksi

### Struktur Folder:

```
ml-practicum/
├── README.md                    # File ini
├── pertemuan-01/
│   ├── README.md               # Materi & tugas pertemuan 1
│   └── [your_work].ipynb       # Notebook hasil praktikum Anda
├── pertemuan-02/
│   ├── README.md
│   └── [your_work].ipynb
├── ...
└── pertemuan-08/
    └── README.md               # Guideline UTS
```

## 💻 Teknologi yang Digunakan

- **Python 3.10+**: Bahasa pemrograman utama
- **NumPy**: Komputasi numerik
- **Pandas**: Data manipulation dan analysis
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning algorithms
- **Jupyter Notebook**: Interactive development environment

## 📊 Sistem Penilaian

| Komponen | Bobot |
|----------|-------|
| Kehadiran & Partisipasi | 10% |
| Tugas Mingguan (Weekly Labs) | 30% |
| UTS (Mid-term Project) | 25% |
| UAS (Final Project & Presentation) | 35% |
| **TOTAL** | **100%** |

### Kriteria Kelulusan:
- Nilai akhir minimal: **60 (D)**
- Kehadiran minimal: **75%** dari total pertemuan
- Mengumpulkan minimal **75%** dari total tugas
- Mengikuti UTS dan UAS

## 📝 Submission Guidelines

### Format File:
```
NIM_Nama_PertemuanXX.ipynb
NIM_Nama_PertemuanXX.pdf
```

Contoh: `105841101521_JohnDoe_Pertemuan01.pdf`

### Checklist Sebelum Submit:
- [ ] Semua cell sudah di-run dan ada output
- [ ] Code rapi dan ada komentar
- [ ] Ada interpretasi untuk setiap hasil
- [ ] Visualisasi jelas dan labeled
- [ ] File naming sesuai format

## 🔧 Troubleshooting

### Error saat import library:
```bash
pip install --upgrade [nama-library]
```

### Jupyter Notebook tidak muncul:
```bash
pip install --upgrade jupyter notebook
jupyter notebook --version
```

### Memory error saat train model:
- Reduce dataset size
- Use `n_jobs=-1` untuk parallel processing
- Close aplikasi lain yang tidak digunakan

## 📚 Referensi & Resources

### Documentation:
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

### Learning Resources:
- [Kaggle Learn](https://www.kaggle.com/learn)
- [Machine Learning Crash Course - Google](https://developers.google.com/machine-learning/crash-course)
- [Scikit-learn Tutorials](https://scikit-learn.org/stable/tutorial/index.html)

### Datasets:
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml)
- [Scikit-learn Built-in Datasets](https://scikit-learn.org/stable/datasets.html)

## 👥 Kontributor

**Dosen Pengampu:**  
[Nama Dosen]

**Asisten Praktikum:**  
[Nama Asisten 1]  
[Nama Asisten 2]

## 📧 Kontak & Support

- **Email:** [email dosen/asisten]
- **Forum Diskusi:** [link ke forum kelas]
- **Office Hours:** [jadwal konsultasi]

## 📄 Lisensi

Materi ini dibuat untuk keperluan pendidikan di Universitas Muhammadiyah Makassar.

---

## 🎓 Tips Sukses

1. **Konsisten:** Kerjakan praktikum setiap minggu, jangan ditumpuk
2. **Praktek:** Machine learning adalah skill yang butuh banyak praktek
3. **Experiment:** Jangan takut untuk coba parameter berbeda
4. **Dokumentasi:** Selalu tulis penjelasan untuk code Anda
5. **Bertanya:** Jika stuck, tanya di forum atau saat konsultasi
6. **Collaborate:** Diskusi dengan teman, tapi jangan plagiat
7. **Real World:** Cari dataset yang Anda minati untuk project

---

## ⚠️ Catatan Penting

- Pastikan selalu **backup** pekerjaan Anda
- **Deadline** adalah hard deadline
- **Plagiarism** akan mendapat sanksi sesuai aturan universitas
- Materi ini akan **terus diupdate**, pastikan pull latest changes

---

## 🚀 Let's Start Learning!

Mulai dari [Pertemuan 01](./pertemuan-01) dan ikuti step-by-step tutorial.

**Happy Learning & Coding! 🎉**

---

**Last Updated:** December 2024  
**Version:** 1.0
