# Pertemuan 8: UTS - Project Mid-semester

## 🎯 Tujuan UTS

Ujian Tengah Semester berbentuk **Mini Project Machine Learning End-to-End** yang bertujuan untuk mengevaluasi pemahaman mahasiswa terhadap:
1. Complete ML pipeline (data preprocessing, modeling, evaluation)
2. Multiple algorithms yang telah dipelajari (Pertemuan 1-7)
3. Analisis dan interpretasi hasil
4. Kemampuan dokumentasi dan presentasi

## 📋 Overview Project

Anda akan membuat **satu project machine learning lengkap** dari awal hingga akhir, mencakup:
- Data loading dan exploratory data analysis
- Data preprocessing dan feature engineering
- Model training dengan minimal 3 algoritma berbeda
- Model evaluation dan comparison
- Dokumentasi dan insight

## 🎓 Pilihan Project

Pilih **SALAH SATU** dari kategori berikut:

### Option 1: Classification Problem
**Dataset suggestions:**
- Titanic Survival Prediction
- Diabetes Prediction
- Heart Disease Prediction
- Credit Card Fraud Detection
- Spam Email Classification

**Requirements:**
- Binary atau multi-class classification
- Minimal 3 algorithms: Logistic Regression, Decision Tree, Random Forest/SVM
- Evaluasi dengan confusion matrix, accuracy, precision, recall, F1-score

### Option 2: Regression Problem
**Dataset suggestions:**
- House Price Prediction
- Car Price Prediction
- Student Performance Prediction
- Salary Prediction

**Requirements:**
- Continuous target variable
- Minimal 3 algorithms: Linear Regression, Decision Tree, Random Forest
- Evaluasi dengan MAE, MSE, RMSE, R² score

### Option 3: Clustering Problem
**Dataset suggestions:**
- Customer Segmentation
- Mall Customer Data
- Wholesale Customer Data

**Requirements:**
- Unsupervised learning
- K-Means dan Hierarchical Clustering
- Elbow method, Silhouette analysis
- Business interpretation per segment

## 📝 Project Structure

### Part 1: Introduction & Data Understanding (15 points)

```markdown
1. Problem Statement
   - Apa problem yang ingin diselesaikan?
   - Mengapa problem ini penting?
   - Apa expected outcome?

2. Dataset Information
   - Source dataset
   - Jumlah samples dan features
   - Deskripsi features
   - Target variable

3. Import Libraries & Load Data
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# ... other imports

# Load dataset
df = pd.read_csv('your_dataset.csv')
print(df.head())
print(df.info())
print(df.describe())
```

### Part 2: Exploratory Data Analysis (20 points)

```python
# 1. Check missing values
print(df.isnull().sum())

# 2. Visualize target distribution
# For classification:
df['target'].value_counts().plot(kind='bar')

# For regression:
df['target'].hist(bins=50)

# 3. Feature correlation
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

# 4. Univariate analysis (minimal 3 features)
# 5. Bivariate analysis (minimal 3 pairs)
# 6. Identify outliers
# 7. Key insights dari EDA
```

**Yang harus ada:**
- Minimal 5 visualisasi berbeda
- Analisis distribusi data
- Correlation analysis
- Outlier detection
- Summary insights (bullet points)

### Part 3: Data Preprocessing (20 points)

```python
# 1. Handling missing values
# 2. Handling outliers (jika perlu)
# 3. Feature encoding (untuk categorical variables)
# 4. Feature scaling/normalization
# 5. Train-test split

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Dokumentasikan:**
- Keputusan preprocessing yang diambil dan alasannya
- Berapa data yang di-drop (jika ada)
- Transformasi yang dilakukan

### Part 4: Model Training & Evaluation (30 points)

```python
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import *

# Dictionary untuk menyimpan models
models = {}
results = []

# Model 1: [Algorithm Name]
model1 = LogisticRegression(random_state=42)
model1.fit(X_train_scaled, y_train)
y_pred1 = model1.predict(X_test_scaled)

# Evaluasi Model 1
# ... metrics

# Model 2: [Algorithm Name]
# ... similar structure

# Model 3: [Algorithm Name]
# ... similar structure

# Comparison table
results_df = pd.DataFrame(results)
print(results_df)
```

**Requirements:**
- **Classification**: Minimal 3 models berbeda
  - Confusion matrix untuk setiap model
  - Accuracy, Precision, Recall, F1-score
  - ROC curve (bonus)
  
- **Regression**: Minimal 3 models berbeda
  - Predicted vs Actual plot
  - MAE, MSE, RMSE, R² score
  - Residual analysis (bonus)

- **Clustering**: K-Means + Hierarchical
  - Elbow method
  - Silhouette analysis
  - Cluster visualization
  - Segment profiling

### Part 5: Hyperparameter Tuning (10 points)

```python
from sklearn.model_selection import GridSearchCV

# Pilih 1 model terbaik untuk tuning
param_grid = {
    'param1': [value1, value2, value3],
    'param2': [value1, value2]
}

grid_search = GridSearchCV(
    best_model,
    param_grid,
    cv=5,
    scoring='accuracy',  # or appropriate metric
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_:.4f}")

# Evaluate tuned model on test set
```

### Part 6: Final Results & Conclusion (15 points)

```markdown
## Final Model Comparison

| Model | Metric1 | Metric2 | Metric3 |
|-------|---------|---------|---------|
| Model A | 0.XX | 0.XX | 0.XX |
| Model B | 0.XX | 0.XX | 0.XX |
| Model C | 0.XX | 0.XX | 0.XX |

## Best Model: [Model Name]

**Why this model performs best:**
- [Reason 1]
- [Reason 2]

## Key Findings:
1. [Finding 1]
2. [Finding 2]
3. [Finding 3]

## Recommendations:
1. [Recommendation 1]
2. [Recommendation 2]

## Future Improvements:
1. [Improvement 1]
2. [Improvement 2]

## Challenges Faced:
1. [Challenge 1 and how you solved it]
2. [Challenge 2 and how you solved it]
```

## 📤 Deliverables

### 1. Jupyter Notebook (.ipynb)
- File name: `NIM_Nama_UTS_MLPracticum.ipynb`
- Harus bisa di-run dari awal sampai akhir tanpa error
- Setiap cell harus ada output yang ter-render
- Gunakan markdown cells untuk dokumentasi

### 2. PDF Export
- Export notebook ke PDF
- File name: `NIM_Nama_UTS_MLPracticum.pdf`
- Pastikan semua visualisasi terlihat jelas

### 3. Dataset
- Include dataset yang digunakan (jika < 5MB)
- Atau provide link download dataset
- Sertakan data source/reference

### 4. README.md (optional, bonus +5)
- Brief project description
- How to run the notebook
- Requirements/dependencies
- Key results summary

## ✅ Rubrik Penilaian

| Komponen | Points | Kriteria |
|----------|--------|----------|
| **Part 1: Introduction** | 15 | Problem statement jelas, dataset well-described |
| **Part 2: EDA** | 20 | Minimal 5 visualisasi, insights mendalam |
| **Part 3: Preprocessing** | 20 | Handling missing values, outliers, scaling, dokumentasi keputusan |
| **Part 4: Modeling** | 30 | Minimal 3 models, evaluasi lengkap, comparison jelas |
| **Part 5: Tuning** | 10 | Grid search implemented, improvement shown |
| **Part 6: Conclusion** | 15 | Interpretasi hasil, recommendations, refleksi |
| **Code Quality** | 10 | Clean code, well-commented, reproducible |
| **Documentation** | 10 | Markdown cells informatif, penjelasan clear |
| **Visualization** | 10 | Grafik informatif, labeled dengan baik |
| **Creativity** | 5 | Feature engineering, insight unik, presentation |
| **TOTAL** | **145** | **Normalized to 100** |

### Breakdown Detail:

**Part 2: EDA (20 points)**
- Missing values analysis (3 pts)
- Target distribution (3 pts)
- Feature correlation (4 pts)
- Univariate analysis (3 pts)
- Bivariate analysis (4 pts)
- Insights summary (3 pts)

**Part 4: Modeling (30 points)**
- Model 1: Implementation & evaluation (8 pts)
- Model 2: Implementation & evaluation (8 pts)
- Model 3: Implementation & evaluation (8 pts)
- Model comparison (6 pts)

**Code Quality (10 points)**
- Naming conventions (2 pts)
- Comments & documentation (3 pts)
- Code organization (2 pts)
- Reproducibility (3 pts)

## 🚫 Apa yang TIDAK Boleh Dilakukan

1. ❌ Copy-paste code tanpa pemahaman
2. ❌ Plagiarism dari internet/teman
3. ❌ Menggunakan dataset yang terlalu simple (< 500 rows atau < 5 features)
4. ❌ Model tanpa evaluasi yang proper
5. ❌ Notebook yang tidak bisa di-run
6. ❌ Tidak ada interpretasi atau insight

## ✅ Best Practices

1. ✅ Pilih dataset yang menarik dan relevant
2. ✅ Dokumentasikan setiap langkah dengan markdown
3. ✅ Visualisasi harus jelas dan informatif
4. ✅ Interpretasi hasil berdasarkan context bisnis/domain
5. ✅ Test notebook dari awal sampai akhir sebelum submit
6. ✅ Gunakan random_state untuk reproducibility

## 💡 Tips Sukses

### Time Management
- **Day 1-2**: Pilih dataset, EDA
- **Day 3-4**: Preprocessing, baseline models
- **Day 5**: Model comparison, tuning
- **Day 6**: Documentation, review
- **Day 7**: Final check, submit

### Dataset Selection
- Kaggle: https://www.kaggle.com/datasets
- UCI ML Repository: https://archive.ics.uci.edu/ml
- Scikit-learn built-in datasets (untuk practice)

### Grading Focus
Points terbesar ada di:
1. **Modeling (30%)**: Pastikan 3 models ter-implement dengan baik
2. **EDA (20%)**: Visualisasi dan insights
3. **Preprocessing (20%)**: Handling data dengan benar

## 📚 Resources

**Documentation:**
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)

**Tutorials:**
- [Kaggle Learn](https://www.kaggle.com/learn)
- [Machine Learning Mastery](https://machinelearningmastery.com/)

**Example Projects:**
- [Kaggle Notebooks](https://www.kaggle.com/notebooks)
- Lihat contoh-contoh dari pertemuan 1-7

## ❓ FAQ

**Q: Boleh pakai dataset dari pertemuan sebelumnya?**
A: Boleh, tapi harus ada analisis yang lebih mendalam dan complete pipeline.

**Q: Berapa minimal jumlah rows dan features?**
A: Minimal 500 rows dan 5 features (tidak termasuk target).

**Q: Boleh pakai library selain scikit-learn?**
A: Boleh (XGBoost, LightGBM), tapi fokus utama tetap di algoritma yang sudah dipelajari.

**Q: Kalau model accuracy rendah, apakah nilai juga rendah?**
A: Tidak. Yang dinilai adalah process, analysis, dan interpretation. Accuracy rendah bisa dijelaskan dan di-improve.

**Q: Boleh konsultasi dengan dosen/asisten?**
A: Boleh untuk klarifikasi assignment, tapi tidak boleh minta dibuatkan code.

## 📧 Submission

**Deadline:** [Sesuai jadwal kuliah - Pertemuan 8]

**Method:**
1. Upload ke LMS
2. Atau push ke GitHub dan share link

**Format:**
```
NIM_Nama_UTS_MLPracticum/
├── NIM_Nama_UTS_MLPracticum.ipynb
├── NIM_Nama_UTS_MLPracticum.pdf
├── dataset.csv (or link.txt)
└── README.md (optional)
```

**Late Submission:**
- 1 day late: -10 points
- 2 days late: -20 points
- 3+ days late: -50 points

---

## 🎯 Final Checklist

Sebelum submit, pastikan:

- [ ] Notebook bisa di-run dari awal sampai akhir
- [ ] Semua cell ada output
- [ ] Minimal 3 models di-implement
- [ ] Evaluasi metrics lengkap
- [ ] Visualisasi minimal 5 buah
- [ ] Ada interpretasi dan insights
- [ ] Markdown cells informatif
- [ ] File naming sesuai format
- [ ] PDF ter-export dengan baik
- [ ] Dataset included atau link provided

---

**Good luck! 🚀**

**Remember:** Ini adalah kesempatan untuk showcase apa yang sudah Anda pelajari di pertemuan 1-7. Tunjukkan best effort Anda!

**Questions?** Contact asisten praktikum atau post di forum kelas.

---

**Selamat mengerjakan UTS! Semoga sukses! 🎓✨**
