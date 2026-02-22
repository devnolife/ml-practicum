"""
Pertemuan 7: K-Means Clustering dan Hierarchical Clustering
Contoh program lengkap untuk unsupervised learning - clustering
"""

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

print("=" * 60)
print("PERTEMUAN 7: Clustering (Unsupervised Learning)")
print("=" * 60)

# ============================================================================
# 1. K-MEANS CLUSTERING dengan Synthetic Data
# ============================================================================
print("\n1. K-MEANS CLUSTERING - Synthetic Data")
print("-" * 60)

# Generate synthetic data dengan 3 clusters
np.random.seed(42)
X_blobs, y_true = make_blobs(n_samples=300, centers=3, n_features=2, 
                             cluster_std=0.6, random_state=42)

print(f"Dataset: {X_blobs.shape[0]} samples, {X_blobs.shape[1]} features")
print(f"True clusters: 3")

# Visualisasi data asli
plt.figure(figsize=(10, 6))
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], s=50, alpha=0.6, edgecolors='black')
plt.title('Original Data (Unlabeled)', fontsize=14, fontweight='bold')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan07_original_data.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualisasi original data disimpan")
plt.close()

# K-Means clustering dengan K=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_blobs)

print("\n--- K-Means Results (K=3) ---")
print(f"Cluster centers:\n{kmeans.cluster_centers_}")
print(f"\nInertia (Within-cluster sum of squares): {kmeans.inertia_:.2f}")
print(f"Silhouette Score: {silhouette_score(X_blobs, clusters):.4f}")

# Visualisasi hasil clustering
plt.figure(figsize=(12, 5))

# Plot 1: Clustering results
plt.subplot(1, 2, 1)
scatter = plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=clusters, 
                     s=50, cmap='viridis', alpha=0.6, edgecolors='black')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
           s=300, c='red', marker='*', edgecolors='black', linewidths=2,
           label='Centroids')
plt.title('K-Means Clustering (K=3)', fontsize=14, fontweight='bold')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)

# Plot 2: True labels (untuk perbandingan)
plt.subplot(1, 2, 2)
scatter = plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_true, 
                     s=50, cmap='viridis', alpha=0.6, edgecolors='black')
plt.title('True Labels', fontsize=14, fontweight='bold')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan07_kmeans_basic.png', dpi=300, bbox_inches='tight')
print("✓ Visualisasi K-Means basic disimpan")
plt.close()

# ============================================================================
# 2. ELBOW METHOD - Menentukan K Optimal
# ============================================================================
print("\n2. ELBOW METHOD - Menentukan K Optimal")
print("-" * 60)

# Coba berbagai nilai K
K_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters_temp = kmeans_temp.fit_predict(X_blobs)
    inertias.append(kmeans_temp.inertia_)
    silhouette_scores.append(silhouette_score(X_blobs, clusters_temp))
    print(f"K={k}: Inertia={kmeans_temp.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.4f}")

# Visualisasi Elbow Method
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Determining Optimal K', fontsize=16, fontweight='bold')

# Plot 1: Elbow Method (Inertia)
axes[0].plot(K_range, inertias, marker='o', linewidth=2, markersize=10)
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia (Within-cluster sum of squares)')
axes[0].set_title('Elbow Method')
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=3, color='red', linestyle='--', label='Optimal K=3')
axes[0].legend()

# Plot 2: Silhouette Score
axes[1].plot(K_range, silhouette_scores, marker='o', linewidth=2, markersize=10, color='green')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score Method')
axes[1].grid(True, alpha=0.3)
axes[1].axvline(x=3, color='red', linestyle='--', label='Optimal K=3')
axes[1].legend()

plt.tight_layout()
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan07_elbow_method.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualisasi Elbow Method disimpan")
plt.close()

optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\nOptimal K based on Silhouette Score: {optimal_k}")

# ============================================================================
# 3. K-MEANS pada Iris Dataset
# ============================================================================
print("\n3. K-MEANS CLUSTERING - Iris Dataset")
print("-" * 60)

# Load Iris dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

print(f"Iris Dataset: {X_iris.shape[0]} samples, {X_iris.shape[1]} features")

# Scaling (penting untuk clustering!)
scaler = StandardScaler()
X_iris_scaled = scaler.fit_transform(X_iris)

# K-Means dengan K=3 (karena ada 3 species)
kmeans_iris = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters_iris = kmeans_iris.fit_predict(X_iris_scaled)

# Evaluasi
silhouette_iris = silhouette_score(X_iris_scaled, clusters_iris)
davies_bouldin_iris = davies_bouldin_score(X_iris_scaled, clusters_iris)

print(f"\nSilhouette Score: {silhouette_iris:.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin_iris:.4f} (lower is better)")

# Cluster distribution
print("\nCluster Distribution:")
unique, counts = np.unique(clusters_iris, return_counts=True)
for cluster, count in zip(unique, counts):
    print(f"  Cluster {cluster}: {count} samples")

# Compare dengan true labels
print("\nComparison with True Labels:")
comparison_df = pd.DataFrame({
    'True Species': [iris.target_names[i] for i in y_iris],
    'Cluster': clusters_iris
})
print(pd.crosstab(comparison_df['True Species'], comparison_df['Cluster'], 
                  margins=True, margins_name='Total'))

# Visualisasi (menggunakan 2 features untuk simplicity)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('K-Means Clustering - Iris Dataset', fontsize=16, fontweight='bold')

# Plot 1: Clustering results
scatter1 = axes[0].scatter(X_iris[:, 0], X_iris[:, 1], c=clusters_iris, 
                          s=50, cmap='viridis', alpha=0.6, edgecolors='black')
axes[0].set_xlabel(iris.feature_names[0])
axes[0].set_ylabel(iris.feature_names[1])
axes[0].set_title(f'K-Means Clusters (Silhouette: {silhouette_iris:.4f})')
plt.colorbar(scatter1, ax=axes[0], label='Cluster')
axes[0].grid(True, alpha=0.3)

# Plot 2: True labels
scatter2 = axes[1].scatter(X_iris[:, 0], X_iris[:, 1], c=y_iris, 
                          s=50, cmap='viridis', alpha=0.6, edgecolors='black')
axes[1].set_xlabel(iris.feature_names[0])
axes[1].set_ylabel(iris.feature_names[1])
axes[1].set_title('True Species Labels')
cbar = plt.colorbar(scatter2, ax=axes[1], label='Species')
cbar.set_ticks([0, 1, 2])
cbar.set_ticklabels(iris.target_names)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan07_kmeans_iris.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualisasi K-Means Iris disimpan")
plt.close()

# ============================================================================
# 4. HIERARCHICAL CLUSTERING
# ============================================================================
print("\n4. HIERARCHICAL CLUSTERING")
print("-" * 60)

# Gunakan subset data untuk visualisasi yang lebih jelas
np.random.seed(42)
sample_indices = np.random.choice(X_iris.shape[0], 50, replace=False)
X_iris_sample = X_iris_scaled[sample_indices]
y_iris_sample = y_iris[sample_indices]

print(f"Sample size: {X_iris_sample.shape[0]} samples")

# Compute linkage matrix
linkage_methods = ['single', 'complete', 'average', 'ward']
linkage_matrices = {}

for method in linkage_methods:
    linkage_matrices[method] = linkage(X_iris_sample, method=method)
    print(f"Linkage computed for method: {method}")

# Visualisasi Dendrograms
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Hierarchical Clustering - Dendrograms', fontsize=16, fontweight='bold')

for idx, method in enumerate(linkage_methods):
    row = idx // 2
    col = idx % 2
    
    dendrogram(linkage_matrices[method], ax=axes[row, col], 
              color_threshold=0, above_threshold_color='black')
    axes[row, col].set_title(f'{method.capitalize()} Linkage', fontsize=12, fontweight='bold')
    axes[row, col].set_xlabel('Sample Index')
    axes[row, col].set_ylabel('Distance')
    axes[row, col].axhline(y=10, color='red', linestyle='--', linewidth=2, label='Cut Height')
    axes[row, col].legend()

plt.tight_layout()
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan07_dendrograms.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualisasi Dendrograms disimpan")
plt.close()

# Agglomerative Clustering dengan Ward linkage
agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
clusters_agg = agg_clustering.fit_predict(X_iris_scaled)

# Evaluasi
silhouette_agg = silhouette_score(X_iris_scaled, clusters_agg)
davies_bouldin_agg = davies_bouldin_score(X_iris_scaled, clusters_agg)

print(f"\n--- Agglomerative Clustering (Ward Linkage) ---")
print(f"Silhouette Score: {silhouette_agg:.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin_agg:.4f}")

# ============================================================================
# 5. CUSTOMER SEGMENTATION EXAMPLE
# ============================================================================
print("\n5. CUSTOMER SEGMENTATION EXAMPLE")
print("-" * 60)

# Generate synthetic customer data
np.random.seed(42)
n_customers = 200

customer_data = pd.DataFrame({
    'Age': np.random.randint(18, 70, n_customers),
    'Income': np.random.randint(20000, 150000, n_customers),
    'Spending_Score': np.random.randint(1, 100, n_customers),
    'Frequency': np.random.randint(1, 50, n_customers)
})

print("\nCustomer Data (5 baris pertama):")
print(customer_data.head())

print("\nStatistik Deskriptif:")
print(customer_data.describe())

# Scaling
scaler_customer = StandardScaler()
customer_scaled = scaler_customer.fit_transform(customer_data)

# Elbow method untuk customer data
inertias_customer = []
silhouette_customer = []
K_range_customer = range(2, 8)

for k in K_range_customer:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters_temp = kmeans_temp.fit_predict(customer_scaled)
    inertias_customer.append(kmeans_temp.inertia_)
    silhouette_customer.append(silhouette_score(customer_scaled, clusters_temp))

# Tentukan K optimal
optimal_k_customer = K_range_customer[np.argmax(silhouette_customer)]
print(f"\nOptimal K for customer segmentation: {optimal_k_customer}")

# Clustering dengan K optimal
kmeans_customer = KMeans(n_clusters=optimal_k_customer, random_state=42, n_init=10)
customer_data['Cluster'] = kmeans_customer.fit_predict(customer_scaled)

print(f"\nCustomer Segmentation Results:")
print(customer_data.groupby('Cluster').agg({
    'Age': 'mean',
    'Income': 'mean',
    'Spending_Score': 'mean',
    'Frequency': 'mean'
}).round(2))

# Visualisasi Customer Segmentation
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Customer Segmentation Analysis', fontsize=16, fontweight='bold')

# Plot 1: Income vs Spending Score
scatter1 = axes[0, 0].scatter(customer_data['Income'], customer_data['Spending_Score'],
                              c=customer_data['Cluster'], s=50, cmap='viridis', 
                              alpha=0.6, edgecolors='black')
axes[0, 0].set_xlabel('Income')
axes[0, 0].set_ylabel('Spending Score')
axes[0, 0].set_title('Income vs Spending Score')
plt.colorbar(scatter1, ax=axes[0, 0], label='Cluster')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Age vs Frequency
scatter2 = axes[0, 1].scatter(customer_data['Age'], customer_data['Frequency'],
                              c=customer_data['Cluster'], s=50, cmap='viridis', 
                              alpha=0.6, edgecolors='black')
axes[0, 1].set_xlabel('Age')
axes[0, 1].set_ylabel('Purchase Frequency')
axes[0, 1].set_title('Age vs Purchase Frequency')
plt.colorbar(scatter2, ax=axes[0, 1], label='Cluster')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Cluster Distribution
cluster_counts = customer_data['Cluster'].value_counts().sort_index()
axes[1, 0].bar(cluster_counts.index, cluster_counts.values, 
               color='skyblue', edgecolor='black')
axes[1, 0].set_xlabel('Cluster')
axes[1, 0].set_ylabel('Number of Customers')
axes[1, 0].set_title('Customer Distribution per Cluster')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Elbow Method
axes[1, 1].plot(K_range_customer, silhouette_customer, marker='o', 
                linewidth=2, markersize=10, color='green')
axes[1, 1].set_xlabel('Number of Clusters (K)')
axes[1, 1].set_ylabel('Silhouette Score')
axes[1, 1].set_title('Optimal K Selection')
axes[1, 1].axvline(x=optimal_k_customer, color='red', linestyle='--', 
                   label=f'Optimal K={optimal_k_customer}')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan07_customer_segmentation.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualisasi Customer Segmentation disimpan")
plt.close()

# Save customer segmentation results
customer_data.to_csv('/workspaces/ml-practicum/contohPengerjaan/customer_segmentation_results.csv', index=False)
print("✓ Customer segmentation results disimpan ke CSV")

# ============================================================================
# 6. COMPARISON: K-Means vs Hierarchical
# ============================================================================
print("\n6. COMPARISON: K-Means vs Hierarchical Clustering")
print("-" * 60)

# Comparison metrics
comparison_data = {
    'Method': ['K-Means', 'Hierarchical (Ward)'],
    'Silhouette Score': [silhouette_iris, silhouette_agg],
    'Davies-Bouldin Index': [davies_bouldin_iris, davies_bouldin_agg]
}
comparison_df = pd.DataFrame(comparison_data)

print("\nPerbandingan Metrik (Iris Dataset):")
print(comparison_df)

# Visualisasi perbandingan
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('K-Means vs Hierarchical Clustering Comparison', fontsize=16, fontweight='bold')

# Plot 1: Silhouette Score
axes[0].bar(comparison_df['Method'], comparison_df['Silhouette Score'], 
            color=['#4ECDC4', '#FF6B6B'], edgecolor='black')
axes[0].set_ylabel('Silhouette Score (Higher is Better)')
axes[0].set_title('Silhouette Score Comparison')
axes[0].grid(True, alpha=0.3, axis='y')

# Plot 2: Davies-Bouldin Index
axes[1].bar(comparison_df['Method'], comparison_df['Davies-Bouldin Index'], 
            color=['#4ECDC4', '#FF6B6B'], edgecolor='black')
axes[1].set_ylabel('Davies-Bouldin Index (Lower is Better)')
axes[1].set_title('Davies-Bouldin Index Comparison')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/workspaces/ml-practicum/contohPengerjaan/pertemuan07_clustering_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualisasi Clustering Comparison disimpan")
plt.close()

print("\n" + "=" * 60)
print("SELESAI - Pertemuan 7")
print("=" * 60)
print("\nRingkasan yang dipelajari:")
print("✓ K-Means Clustering")
print("✓ Elbow Method dan Silhouette Score")
print("✓ Hierarchical Clustering (Dendrogram)")
print("✓ Different Linkage Methods")
print("✓ Customer Segmentation")
print("✓ Clustering Evaluation Metrics")
print("✓ K-Means vs Hierarchical Comparison")
print("\nFile yang dibuat:")
print("- pertemuan07_original_data.png")
print("- pertemuan07_kmeans_basic.png")
print("- pertemuan07_elbow_method.png")
print("- pertemuan07_kmeans_iris.png")
print("- pertemuan07_dendrograms.png")
print("- pertemuan07_customer_segmentation.png")
print("- pertemuan07_clustering_comparison.png")
print("- customer_segmentation_results.csv")
