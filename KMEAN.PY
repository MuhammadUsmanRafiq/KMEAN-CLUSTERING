import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# Generate synthetic data for customer segmentation
data, true_labels = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply K-means clustering
k = 3 # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
predicted_labels = kmeans.fit_predict(data_scaled)
centroids = kmeans.cluster_centers_

# Visualize the results
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=predicted_labels, cmap='viridis', edgecolors='k')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, linewidths=3, color='red', label='Centroids')
plt.title('K-Means Clustering for Customer Segmentation')
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Feature 2 (Standardized)')
plt.legend()
plt.show()