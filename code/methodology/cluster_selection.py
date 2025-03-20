import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering

# Set global font size for all elements
plt.rcParams.update({
    "font.size": 8,  # Adjust font size for all elements
    "axes.titlesize": 10,  # Title font size
    "axes.labelsize": 8,  # X and Y axis label font size
    "legend.fontsize": 7,  # Legend font size
    "xtick.labelsize": 7,  # X-axis tick labels
    "ytick.labelsize": 7,  # Y-axis tick labels
})

# Define datasets separately for Region 1 and Region 2
data1 = pd.DataFrame({
    "Year": [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017],
    "X1": [5, 25, 50, 100, 180, 220, 300, 350],
    "X2": [60, 55, 40, 20, 15, 10, 80, 130],
    "X3": [5, 15, 35, 75, 55, 45, 25, 10],
    "X4": [2, 5, 10, 20, 30, 60, 70, 90],
    "X5": [110, 90, 70, 50, 30, 15, 130, 160],
    "X6": [8, 20, 45, 25, 60, 35, 90, 50],
})

data2 = pd.DataFrame({
    "Year": [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017],
    "T1": [10, 30, 50, 40, 60, 70, 90, 80],
    "T2": [5, 20, 15, 35, 25, 50, 45, 60],
    "T3": [10, 25, 40, 30, 50, 60, 75, 90],
    "T4": [3, 10, 20, 15, 25, 40, 35, 55],
    "T5": [7, 18, 12, 28, 22, 35, 30, 45],
    "T6": [2, 8, 15, 25, 20, 40, 35, 55],
})

# Extract numerical data for clustering (excluding Year)
region1_data = data1.drop(columns=['Year']).values
region2_data = data2.drop(columns=['Year']).values

# Define range for cluster analysis
range_clusters = range(2, 6)

# Hierarchical Clustering for Region 1
linkage_matrix_r1 = linkage(region1_data, method='ward')

# Compute silhouette scores for Region 1
silhouette_scores_r1 = []
for k in range_clusters:
    cluster_labels = fcluster(linkage_matrix_r1, k, criterion='maxclust')
    silhouette_scores_r1.append(silhouette_score(region1_data, cluster_labels))

# Compute compactness for Region 1
compactness_scores_r1 = []
for k in range_clusters:
    clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = clustering.fit_predict(region1_data)
    cluster_centers = [region1_data[labels == i].mean(axis=0) for i in range(k)]
    wcss = sum(np.linalg.norm(region1_data[labels == i] - center) ** 2 for i, center in enumerate(cluster_centers))
    compactness_scores_r1.append(wcss)

# Hierarchical Clustering for Region 2
linkage_matrix_r2 = linkage(region2_data, method='ward')

# Compute silhouette scores for Region 2
silhouette_scores_r2 = []
for k in range_clusters:
    cluster_labels = fcluster(linkage_matrix_r2, k, criterion='maxclust')
    silhouette_scores_r2.append(silhouette_score(region2_data, cluster_labels))

# Compute compactness for Region 2
compactness_scores_r2 = []
for k in range_clusters:
    clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = clustering.fit_predict(region2_data)
    cluster_centers = [region2_data[labels == i].mean(axis=0) for i in range(k)]
    wcss = sum(np.linalg.norm(region2_data[labels == i] - center) ** 2 for i, center in enumerate(cluster_centers))
    compactness_scores_r2.append(wcss)

# Create side-by-side plots for both regions including Silhouette Scores and Compactness
fig, axes = plt.subplots(3, 2, figsize=(8, 9))

# Dendrograms
axes[0, 0].set_title("Dendrogram for Region 1")
sch.dendrogram(linkage_matrix_r1, ax=axes[0, 0])
axes[0, 0].axhline(y=100, color='r', linestyle='--', label="Cut-Off")
axes[0, 0].set_xlabel("Data Points")
axes[0, 0].set_ylabel("Distance")
axes[0, 0].legend()

axes[0, 1].set_title("Dendrogram for Region 2")
sch.dendrogram(linkage_matrix_r2, ax=axes[0, 1])
axes[0, 1].axhline(y=100, color='r', linestyle='--', label="Cut-Off")
axes[0, 1].set_xlabel("Data Points")
axes[0, 1].set_ylabel("Distance")
axes[0, 1].legend()

# Compactness Scores
axes[1, 0].plot(range_clusters, compactness_scores_r1, marker='s', linestyle='-', color='b', label="Region 1")
axes[1, 0].set_xlabel("Number of Clusters")
axes[1, 0].set_ylabel("Compactness (WCSS)")
axes[1, 0].set_title("Compactness Score (Region 1)")
axes[1, 0].grid(True)
axes[1, 0].legend()

axes[1, 1].plot(range_clusters, compactness_scores_r2, marker='s', linestyle='-', color='g', label="Region 2")
axes[1, 1].set_xlabel("Number of Clusters")
axes[1, 1].set_ylabel("Compactness (WCSS)")
axes[1, 1].set_title("Compactness Score (Region 2)")
axes[1, 1].grid(True)
axes[1, 1].legend()

# Silhouette Scores
axes[2, 0].plot(range_clusters, silhouette_scores_r1, marker='o', linestyle='-', color='b', label="Region 1")
axes[2, 0].set_xlabel("Number of Clusters")
axes[2, 0].set_ylabel("Silhouette Score")
axes[2, 0].set_title("Silhouette Score (Region 1)")
axes[2, 0].grid(True)
axes[2, 0].legend()

axes[2, 1].plot(range_clusters, silhouette_scores_r2, marker='o', linestyle='-', color='g', label="Region 2")
axes[2, 1].set_xlabel("Number of Clusters")
axes[2, 1].set_ylabel("Silhouette Score")
axes[2, 1].set_title("Silhouette Score (Region 2)")
axes[2, 1].grid(True)
axes[2, 1].legend()

plt.tight_layout()
plt.show()