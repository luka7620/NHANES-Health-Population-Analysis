import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

def spectral_clustering(scaled_data, n_clusters=5):
    """谱聚类算法实现"""
    print(f"正在进行谱聚类 (k={n_clusters})...")
    start_time = time.time()
    spectral = SpectralClustering(n_clusters=n_clusters, random_state=42, affinity="nearest_neighbors")
    clusters = spectral.fit_predict(scaled_data)
    end_time = time.time()
    silhouette_avg = silhouette_score(scaled_data, clusters)
    print(f"谱聚类轮廓系数: {silhouette_avg:.4f}")
    print(f"谱聚类用时: {end_time - start_time:.2f}秒")
    return clusters 