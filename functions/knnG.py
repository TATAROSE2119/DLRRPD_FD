import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_knn_graph(X, k):
    """
    构建k最近邻图
    X: 训练数据，形状为 (n_samples, n_features)
    k: 每个样本的邻居数
    返回: k最近邻图（相似度矩阵）
    """
    # 使用 sklearn NearestNeighbors 计算 k 最近邻
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(X)

    # 找到每个样本的 k 个最近邻
    distances, indices = nn.kneighbors(X)

    # 初始化相似度矩阵 Z，大小为 (n_samples, n_samples)
    n_samples = X.shape[1]
    Z = np.zeros((n_samples, n_samples))

    # 填充 Z 矩阵
    for i in range(n_samples):
        for j in range(k):
            # 赋值为相似度，通常这里可以使用距离的倒数或相似度度量
            Z[i, indices[i, j]] = np.exp(-distances[i, j] ** 2)  # 采用高斯相似度
            Z[indices[i, j], i] = Z[i, indices[i, j]]  # 对称矩阵

    return Z
