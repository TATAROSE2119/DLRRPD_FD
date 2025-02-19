import numpy as np
from sklearn.neighbors import kneighbors_graph


def compute_knn_graph(X, k):
    """
    初始化Z矩阵，使用k近邻图
    X: 数据矩阵，每一行是一个样本
    k: 每个样本的邻居数
    返回：Z矩阵（k近邻图的相似度矩阵）
    """
    # 计算k近邻图
    X=X.T

    Z = kneighbors_graph(X, n_neighbors=k, mode='connectivity', include_self=False)

    # 将稀疏矩阵转换为密集矩阵
    Z = Z.toarray()

    # 确保Z的对角线元素为0
    np.fill_diagonal(Z, 0)

    return Z
