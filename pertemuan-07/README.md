# Pertemuan 7: K-Means Clustering dan Hierarchical Clustering

## 🎯 Tujuan Pembelajaran

Setelah menyelesaikan pertemuan ini, mahasiswa diharapkan mampu:
1. Memahami konsep unsupervised learning dan clustering
2. Mengimplementasikan K-Means Clustering
3. Menentukan jumlah cluster optimal dengan Elbow Method dan Silhouette Score
4. Mengimplementasikan Hierarchical Clustering
5. Membuat dan menginterpretasi dendrogram
6. Melakukan customer segmentation menggunakan clustering

## 📚 Teori Singkat

### Unsupervised Learning vs Supervised Learning

| Aspect | Supervised | Unsupervised |
|--------|-----------|--------------|
| Labels | Ada (y) | Tidak ada |
| Tujuan | Predict labels | Find patterns/structure |
| Contoh | Classification, Regression | Clustering, Dimensionality Reduction |
| Evaluasi | Accuracy, RMSE | Silhouette, Within-cluster variance |

### K-Means Clustering

**Algoritma:**
1. Pilih K (jumlah cluster)
2. Initialize K centroids secara random
3. Assign setiap data point ke centroid terdekat
4. Update centroid (rata-rata dari data points dalam cluster)
5. Repeat step 3-4 sampai converge

**Karakteristik:**
- Fast dan scalable
- Butuh specify K di awal
- Sensitive terhadap initialization dan outliers
- Assume clusters berbentuk spherical

**Metrics:**
- **Inertia (Within-cluster sum of squares)**: Semakin kecil semakin baik
- **Silhouette Score**: -1 to 1, semakin tinggi semakin baik

### Hierarchical Clustering

**Jenis:**
1. **Agglomerative (Bottom-up)**:
   - Start: Setiap data point adalah cluster sendiri
   - Iteratively merge clusters terdekat
   - End: Semua data dalam 1 cluster

2. **Divisive (Top-down)**:
   - Start: Semua data dalam 1 cluster
   - Iteratively split clusters
   - End: Setiap data point adalah cluster sendiri

**Linkage Methods:**
- **Single**: Minimum distance antara points dari 2 clusters
- **Complete**: Maximum distance
- **Average**: Average distance
- **Ward**: Minimize within-cluster variance

**Visualisasi**: Dendrogram

## 📝 Praktikum

### Persiapan: Import Library

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.datasets import make_blobs, load_iris
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# Settings
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
```

### Langkah 1: K-Means dengan Synthetic Data

```python
# Generate synthetic data dengan 4 clusters
X, y_true = make_blobs(
    n_samples=300,
    centers=4,
    n_features=2,
    cluster_std=0.6,
    random_state=42
)

# Visualisasi data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.6, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Synthetic Dataset for Clustering')
plt.show()

# Apply K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
y_pred = kmeans.fit_predict(X)

# Visualisasi hasil clustering
plt.figure(figsize=(12, 5))

# Ground truth
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Ground Truth (4 clusters)')
plt.colorbar(label='Cluster')

# K-Means result
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6, edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:, 0], 
           kmeans.cluster_centers_[:, 1],
           marker='X', s=300, c='red', edgecolors='black', linewidths=2,
           label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering Result')
plt.legend()
plt.colorbar(label='Cluster')

plt.tight_layout()
plt.show()

# Evaluasi
print("=== K-Means Evaluation ===")
print(f"Inertia (Within-cluster sum of squares): {kmeans.inertia_:.2f}")
print(f"Silhouette Score: {silhouette_score(X, y_pred):.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin_score(X, y_pred):.4f}")
```

### Langkah 2: Elbow Method - Menentukan K Optimal

```python
# Test berbagai nilai K
K_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42)
    labels = kmeans_temp.fit_predict(X)
    
    inertias.append(kmeans_temp.inertia_)
    silhouette_scores.append(silhouette_score(X, labels))

# Visualisasi Elbow Method
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Elbow curve (Inertia)
axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method - Inertia')
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=4, color='r', linestyle='--', label='Optimal K=4')
axes[0].legend()

# Silhouette Score
axes[1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score vs K')
axes[1].grid(True, alpha=0.3)
axes[1].axvline(x=4, color='r', linestyle='--', label='Optimal K=4')
axes[1].legend()

plt.tight_layout()
plt.show()

print("\n=== Optimal K Analysis ===")
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"K dengan Silhouette Score tertinggi: {optimal_k}")
print(f"Silhouette Score: {max(silhouette_scores):.4f}")
```

### Langkah 3: Visualisasi per Cluster

```python
# K-Means dengan K optimal
kmeans_final = KMeans(n_clusters=4, random_state=42)
clusters = kmeans_final.fit_predict(X)

# Analisis per cluster
df_clustered = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
df_clustered['Cluster'] = clusters

print("=== Cluster Statistics ===")
print(df_clustered.groupby('Cluster').describe())

# Visualisasi distribusi per cluster
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for cluster_id in range(4):
    row = cluster_id // 2
    col = cluster_id % 2
    
    cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
    
    axes[row, col].scatter(cluster_data['Feature1'], cluster_data['Feature2'],
                          alpha=0.6, edgecolors='k')
    axes[row, col].scatter(kmeans_final.cluster_centers_[cluster_id, 0],
                          kmeans_final.cluster_centers_[cluster_id, 1],
                          marker='X', s=300, c='red', edgecolors='black', linewidths=2)
    axes[row, col].set_xlabel('Feature 1')
    axes[row, col].set_ylabel('Feature 2')
    axes[row, col].set_title(f'Cluster {cluster_id} (n={len(cluster_data)})')

plt.tight_layout()
plt.show()
```

### Langkah 4: Hierarchical Clustering

```python
# Generate data
X_hier, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

# Compute linkage matrix
linkage_matrix = linkage(X_hier, method='ward')

# Plot dendrogram
plt.figure(figsize=(14, 6))
dendrogram(linkage_matrix, truncate_mode='lastp', p=30, show_leaf_counts=True)
plt.xlabel('Sample Index or (Cluster Size)')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)')
plt.axhline(y=10, color='r', linestyle='--', label='Cut-off threshold')
plt.legend()
plt.show()

# Apply Agglomerative Clustering
n_clusters = 3
agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
labels_agg = agg_clustering.fit_predict(X_hier)

# Visualisasi hasil
plt.figure(figsize=(8, 6))
plt.scatter(X_hier[:, 0], X_hier[:, 1], c=labels_agg, cmap='viridis', 
           alpha=0.6, edgecolors='k', s=50)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'Hierarchical Clustering (n_clusters={n_clusters})')
plt.colorbar(label='Cluster')
plt.show()

print("=== Hierarchical Clustering ===")
print(f"Silhouette Score: {silhouette_score(X_hier, labels_agg):.4f}")
```

### Langkah 5: Comparison of Linkage Methods

```python
# Compare different linkage methods
linkage_methods = ['single', 'complete', 'average', 'ward']
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

for idx, method in enumerate(linkage_methods):
    row = idx // 2
    col = idx % 2
    
    # Hierarchical clustering
    agg = AgglomerativeClustering(n_clusters=3, linkage=method)
    labels = agg.fit_predict(X_hier)
    
    # Silhouette score
    sil_score = silhouette_score(X_hier, labels)
    
    # Plot
    axes[row, col].scatter(X_hier[:, 0], X_hier[:, 1], c=labels, 
                          cmap='viridis', alpha=0.6, edgecolors='k')
    axes[row, col].set_xlabel('Feature 1')
    axes[row, col].set_ylabel('Feature 2')
    axes[row, col].set_title(f'{method.capitalize()} Linkage (Silhouette: {sil_score:.3f})')

plt.tight_layout()
plt.show()

print("\n=== Linkage Methods Comparison ===")
print("Single: Sensitive to noise and outliers")
print("Complete: Less sensitive to outliers, compact clusters")
print("Average: Balance between single and complete")
print("Ward: Minimize within-cluster variance, most commonly used")
```

### Langkah 6: Customer Segmentation

```python
# Generate customer data
np.random.seed(42)
n_customers = 500

customer_data = pd.DataFrame({
    'Age': np.random.randint(18, 70, n_customers),
    'Income': np.random.randint(20000, 150000, n_customers),
    'SpendingScore': np.random.randint(1, 100, n_customers),
    'Visits_per_month': np.random.randint(1, 30, n_customers)
})

print("=== Customer Data ===")
print(customer_data.head(10))
print(f"\nShape: {customer_data.shape}")
print(f"\nStatistics:")
print(customer_data.describe())

# Standardize features
scaler = StandardScaler()
customer_scaled = scaler.fit_transform(customer_data)

# Determine optimal K
K_range = range(2, 11)
inertias = []
silhouettes = []

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42)
    labels = kmeans_temp.fit_predict(customer_scaled)
    inertias.append(kmeans_temp.inertia_)
    silhouettes.append(silhouette_score(customer_scaled, labels))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].plot(K_range, inertias, 'bo-', linewidth=2)
axes[0].set_xlabel('Number of Clusters')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method - Customer Segmentation')
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, silhouettes, 'go-', linewidth=2)
axes[1].set_xlabel('Number of Clusters')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score - Customer Segmentation')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Apply K-Means dengan K optimal
optimal_k = 4  # Based on elbow method
kmeans_customer = KMeans(n_clusters=optimal_k, random_state=42)
customer_data['Segment'] = kmeans_customer.fit_predict(customer_scaled)

# Analisis segment
print("\n=== Customer Segments ===")
segment_profile = customer_data.groupby('Segment').mean()
print(segment_profile)

# Visualisasi segments
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Age vs Income
axes[0, 0].scatter(customer_data['Age'], customer_data['Income'], 
                  c=customer_data['Segment'], cmap='viridis', alpha=0.6, edgecolors='k')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Income')
axes[0, 0].set_title('Age vs Income')

# Income vs Spending Score
axes[0, 1].scatter(customer_data['Income'], customer_data['SpendingScore'],
                  c=customer_data['Segment'], cmap='viridis', alpha=0.6, edgecolors='k')
axes[0, 1].set_xlabel('Income')
axes[0, 1].set_ylabel('Spending Score')
axes[0, 1].set_title('Income vs Spending Score')

# Spending Score vs Visits
axes[0, 2].scatter(customer_data['SpendingScore'], customer_data['Visits_per_month'],
                  c=customer_data['Segment'], cmap='viridis', alpha=0.6, edgecolors='k')
axes[0, 2].set_xlabel('Spending Score')
axes[0, 2].set_ylabel('Visits per Month')
axes[0, 2].set_title('Spending Score vs Visits')

# Segment size
segment_sizes = customer_data['Segment'].value_counts().sort_index()
axes[1, 0].bar(segment_sizes.index, segment_sizes.values, color='skyblue', edgecolor='black')
axes[1, 0].set_xlabel('Segment')
axes[1, 0].set_ylabel('Number of Customers')
axes[1, 0].set_title('Segment Sizes')

# Average characteristics per segment
segment_profile.plot(kind='bar', ax=axes[1, 1])
axes[1, 1].set_xlabel('Segment')
axes[1, 1].set_ylabel('Average Value')
axes[1, 1].set_title('Average Characteristics per Segment')
axes[1, 1].legend(loc='upper right', fontsize=8)
axes[1, 1].tick_params(axis='x', rotation=0)

# Remove extra subplot
fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.show()

# Business interpretation
print("\n=== Business Interpretation ===")
for segment in range(optimal_k):
    segment_data = customer_data[customer_data['Segment'] == segment]
    print(f"\nSegment {segment}:")
    print(f"  Size: {len(segment_data)} customers")
    print(f"  Avg Age: {segment_data['Age'].mean():.1f}")
    print(f"  Avg Income: ${segment_data['Income'].mean():,.0f}")
    print(f"  Avg Spending Score: {segment_data['SpendingScore'].mean():.1f}")
    print(f"  Avg Visits: {segment_data['Visits_per_month'].mean():.1f}")
```

## 💪 Tugas Praktikum

### Tugas 1: K-Means Implementation (25 poin)

Gunakan Iris dataset:
1. Load dataset dan pilih 2 features (untuk visualisasi)
2. Implementasikan Elbow Method untuk K = 2 sampai 10
3. Plot Inertia dan Silhouette Score
4. Tentukan K optimal
5. Visualisasi hasil clustering dengan scatter plot
6. Bandingkan dengan true labels (jika ada)
7. Interpretasi: Apakah K-Means berhasil menemukan struktur natural dari data?

### Tugas 2: Hierarchical Clustering (25 poin)

Gunakan dataset yang sama:
1. Compute linkage matrix dengan 4 methods (single, complete, average, ward)
2. Buat dendrogram untuk setiap method
3. Apply Agglomerative Clustering dengan berbagai n_clusters
4. Bandingkan hasil dengan silhouette score
5. Visualisasi hasil untuk setiap linkage method
6. Kesimpulan: Linkage method mana yang terbaik? Mengapa?

### Tugas 3: K-Means vs Hierarchical (25 poin)

Buat atau download dataset clustering:
1. Apply K-Means
2. Apply Hierarchical Clustering (Ward linkage)
3. Bandingkan:
   - Silhouette score
   - Davies-Bouldin index
   - Visualisasi cluster assignment
   - Computation time
4. Analisis kelebihan dan kekurangan masing-masing
5. Kapan sebaiknya pakai K-Means? Kapan Hierarchical?

### Tugas 4: Customer Segmentation Project (25 poin)

Download dataset customer (contoh: Mall Customer Segmentation dari Kaggle):
1. EDA: Distribusi features, correlation
2. Feature engineering jika perlu
3. Standardize features
4. Determine optimal K dengan multiple methods (Elbow, Silhouette)
5. Apply clustering
6. Profiling setiap segment (mean, median, distribution)
7. Visualisasi dengan multiple scatter plots
8. **Business Recommendations**:
   - Karakteristik setiap segment
   - Marketing strategy untuk setiap segment
   - Which segment is most valuable?
   - Which segment needs attention?

## 📤 Cara Mengumpulkan

1. Notebook lengkap dengan interpretasi bisnis
2. Setiap visualisasi harus ada penjelasan
3. Export ke PDF: `NIM_Nama_Pertemuan07.pdf`
4. Upload ke LMS atau GitHub

## ✅ Kriteria Penilaian

| Aspek | Bobot |
|-------|-------|
| Tugas 1: K-Means | 25% |
| Tugas 2: Hierarchical | 25% |
| Tugas 3: Comparison | 25% |
| Tugas 4: Customer Segmentation | 25% |
| Elbow method & optimal K | 15% |
| Business interpretation | 15% |
| Dokumentasi | 10% |

## 📚 Referensi

1. [K-Means - Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means)
2. [Hierarchical Clustering - Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)
3. [Silhouette Analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)
4. [Customer Segmentation with Clustering](https://towardsdatascience.com/customer-segmentation-using-k-means-clustering-d33964f238c3)

## 💡 Tips

- **Always standardize** features untuk clustering (karena distance-based)
- **Elbow Method** kadang ambiguous, combine dengan Silhouette Score
- **K-Means**: Fast, tapi butuh specify K dan sensitive to outliers
- **Hierarchical**: Tidak perlu specify K diawal, tapi slower untuk large datasets
- **Customer Segmentation**: Focus on business interpretation, bukan hanya accuracy!

---

**Happy Clustering! 🎯📊**
