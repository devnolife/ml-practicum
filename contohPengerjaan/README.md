# Contoh Pengerjaan - ML Practicum

Folder ini berisi contoh program lengkap untuk setiap pertemuan dalam mata kuliah Machine Learning Practicum.

## 📚 Daftar Isi

### Pertemuan 1: Pengenalan Python untuk Machine Learning
**File:** `pertemuan01_python_basics.py`

**Materi:**
- NumPy: Array operations, mathematical functions
- Pandas: DataFrame manipulation, filtering, grouping
- Matplotlib: Data visualization (line plots, bar charts, scatter plots, histograms)

**Cara Menjalankan:**
```bash
python pertemuan01_python_basics.py
```

**Output:**
- `pertemuan01_visualisasi.png` - Visualisasi data penjualan
- `pertemuan01_histogram.png` - Distribusi nilai mahasiswa

---

### Pertemuan 2: Data Cleaning dan EDA
**File:** `pertemuan02_data_cleaning.py`

**Materi:**
- Load dan inspeksi data (Titanic dataset)
- Handling missing values (median, mode, drop)
- Deteksi dan handling outliers (IQR method)
- Feature scaling (StandardScaler, MinMaxScaler)
- Exploratory Data Analysis

**Cara Menjalankan:**
```bash
python pertemuan02_data_cleaning.py
```

**Output:**
- `pertemuan02_missing_values.png` - Heatmap missing values
- `pertemuan02_outliers_boxplot.png` - Boxplot untuk deteksi outliers
- `pertemuan02_scaling_comparison.png` - Perbandingan scaling methods
- `pertemuan02_eda_analysis.png` - Analisis survival Titanic
- `pertemuan02_correlation_heatmap.png` - Correlation matrix
- `titanic_cleaned.csv` - Data yang sudah dibersihkan

---

### Pertemuan 3: Linear Regression
**File:** `pertemuan03_linear_regression.py`

**Materi:**
- Simple Linear Regression (1 variabel)
- Multiple Linear Regression (banyak variabel)
- Polynomial Regression (data non-linear)
- Model evaluation (R², RMSE, MAE)
- Residual analysis

**Cara Menjalankan:**
```bash
python pertemuan03_linear_regression.py
```

**Output:**
- `pertemuan03_simple_regression.png` - Simple linear regression
- `pertemuan03_multiple_regression.png` - Multiple regression results
- `pertemuan03_residual_plot.png` - Residual analysis
- `pertemuan03_polynomial_regression.png` - Polynomial vs linear
- `pertemuan03_polynomial_degrees.png` - Berbagai derajat polynomial

---

### Pertemuan 4: Classification
**File:** `pertemuan04_classification.py`

**Materi:**
- Binary Classification dengan Logistic Regression
- Multi-class Classification (Iris dataset)
- Decision Tree Classifier
- Confusion Matrix dan metrics (Accuracy, Precision, Recall, F1)
- ROC Curve dan AUC
- Feature Importance

**Cara Menjalankan:**
```bash
python pertemuan04_classification.py
```

**Output:**
- `pertemuan04_binary_classification.png` - Binary classification results
- `pertemuan04_multiclass_classification.png` - Multi-class results
- `pertemuan04_decision_tree_viz.png` - Decision tree visualization
- `pertemuan04_model_comparison.png` - Model comparison

---

### Pertemuan 5: Support Vector Machine (SVM)
**File:** `pertemuan05_svm.py`

**Materi:**
- Linear SVM untuk data linearly separable
- Non-linear SVM dengan RBF kernel
- Perbandingan berbagai kernel (linear, poly, rbf, sigmoid)
- Hyperparameter tuning dengan GridSearchCV
- Pengaruh parameter C dan gamma
- Support vectors visualization

**Cara Menjalankan:**
```bash
python pertemuan05_svm.py
```

**Output:**
- `pertemuan05_linear_svm.png` - Linear SVM dengan decision boundary
- `pertemuan05_rbf_kernel.png` - RBF kernel comparison
- `pertemuan05_kernel_comparison.png` - Perbandingan semua kernel
- `pertemuan05_hyperparameter_tuning.png` - Grid search results
- `pertemuan05_c_parameter_effect.png` - Pengaruh parameter C

---

### Pertemuan 6: Ensemble Methods
**File:** `pertemuan06_ensemble_methods.py`

**Materi:**
- Random Forest Classification dan Regression
- Gradient Boosting
- Feature Importance Analysis
- Model Comparison (Bagging vs Boosting)
- Hyperparameter Tuning dengan GridSearchCV
- Cross-Validation

**Cara Menjalankan:**
```bash
python pertemuan06_ensemble_methods.py
```

**Output:**
- `pertemuan06_random_forest_classification.png` - RF classification
- `pertemuan06_random_forest_regression.png` - RF regression
- `pertemuan06_gradient_boosting.png` - Gradient boosting comparison
- `pertemuan06_hyperparameter_tuning.png` - Tuning results

---

### Pertemuan 7: Clustering
**File:** `pertemuan07_clustering.py`

**Materi:**
- K-Means Clustering
- Elbow Method dan Silhouette Score
- Hierarchical Clustering (Dendrogram)
- Different Linkage Methods
- Customer Segmentation
- Clustering Evaluation Metrics

**Cara Menjalankan:**
```bash
python pertemuan07_clustering.py
```

**Output:**
- `pertemuan07_original_data.png` - Data tanpa label
- `pertemuan07_kmeans_basic.png` - K-Means clustering results
- `pertemuan07_elbow_method.png` - Elbow method analysis
- `pertemuan07_kmeans_iris.png` - K-Means pada Iris dataset
- `pertemuan07_dendrograms.png` - Hierarchical clustering
- `pertemuan07_customer_segmentation.png` - Customer segmentation
- `pertemuan07_clustering_comparison.png` - K-Means vs Hierarchical
- `customer_segmentation_results.csv` - Hasil segmentasi

---

### Pertemuan 8: UTS Project
**File:** `pertemuan08_uts_project.py`

**Materi:**
Complete End-to-End Machine Learning Project:
1. Data Loading & Exploration
2. Data Preprocessing & Cleaning
3. Feature Engineering
4. Model Training (5 algorithms)
5. Model Evaluation & Comparison
6. Hyperparameter Tuning
7. Final Model Selection

**Project:** Heart Disease Prediction (Binary Classification)

**Cara Menjalankan:**
```bash
python pertemuan08_uts_project.py
```

**Output:**
- `pertemuan08_eda.png` - Exploratory data analysis
- `pertemuan08_model_evaluation.png` - Model evaluation & comparison
- `model_comparison_results.csv` - Comparison metrics

---

## 🔧 Requirements

Install semua dependencies yang diperlukan:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter
```

Atau jika menggunakan Anaconda:

```bash
conda install numpy pandas matplotlib seaborn scikit-learn scipy jupyter
```

## 📝 Catatan Penting

1. **Dataset**: Beberapa program menggunakan dataset built-in dari scikit-learn (Iris, Titanic, Breast Cancer, Diabetes), dan beberapa menggunakan synthetic data yang di-generate.

2. **Visualisasi**: Semua program akan membuat visualisasi yang disimpan sebagai file PNG. Plot akan muncul di layar dan otomatis tersimpan.

3. **Random Seed**: Semua program menggunakan `random_seed=42` untuk reproducibility.

4. **Output**: Setiap program akan mencetak informasi detail ke console dan menyimpan visualisasi ke file.

## 🚀 Tips Penggunaan

1. **Jalankan program secara berurutan** (pertemuan 1 → 8) untuk memahami progression dari konsep basic hingga advanced.

2. **Baca output console** untuk memahami proses yang terjadi dan hasil evaluasi.

3. **Lihat visualisasi yang dihasilkan** untuk insight visual dari data dan hasil model.

4. **Modify parameters** dalam kode untuk eksperimen dan belajar lebih dalam.

5. **Gabungkan konsep** dari beberapa pertemuan untuk project yang lebih kompleks.

## 📊 Struktur Program

Setiap program mengikuti struktur yang konsisten:

1. **Import Libraries** - Import semua library yang dibutuhkan
2. **Load Data** - Load atau generate dataset
3. **Data Exploration** - Eksplorasi dan visualisasi data
4. **Preprocessing** (jika diperlukan) - Cleaning, scaling, etc.
5. **Model Training** - Train model dengan algoritma yang sesuai
6. **Evaluation** - Evaluate performa model
7. **Visualization** - Visualisasi hasil
8. **Summary** - Ringkasan pembelajaran

## 🎓 Pembelajaran

Setelah menyelesaikan semua program, Anda akan memahami:

- ✅ Python basics untuk Machine Learning
- ✅ Data preprocessing dan cleaning
- ✅ Exploratory Data Analysis (EDA)
- ✅ Regression algorithms
- ✅ Classification algorithms
- ✅ Support Vector Machine
- ✅ Ensemble methods
- ✅ Clustering (Unsupervised Learning)
- ✅ Complete end-to-end ML project workflow

## ❓ Troubleshooting

**Problem**: `ModuleNotFoundError`
**Solution**: Install missing library dengan `pip install <library-name>`

**Problem**: Plot tidak muncul
**Solution**: Pastikan backend matplotlib sudah di-set dengan benar. Tambahkan `plt.show()` jika diperlukan.

**Problem**: Memory error pada grid search
**Solution**: Kurangi parameter grid atau gunakan `n_jobs=1` instead of `-1`

## 📧 Kontak

Jika ada pertanyaan atau issue, silakan hubungi instruktur atau buka issue di repository.

---

**Happy Learning! 🎉**
