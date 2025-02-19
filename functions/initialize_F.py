import numpy as np
from scipy.linalg import eigh

def initialize_F(Z, k):
    """
    根据公式20初始化矩阵 F
    Z: 相似度矩阵
    k: 类别数量
    返回: 初始化后的 F 矩阵
    """
    # 计算拉普拉斯矩阵 L_Z
    WZ = (Z + Z.T) / 2  # 保证相似度矩阵是对称的
    D = np.diag(np.sum(WZ, axis=1))  # 度矩阵
    LZ = D - WZ  # 拉普拉斯矩阵 L_Z

    # 特征分解，得到拉普拉斯矩阵 L_Z 的前 k 个特征向量
    eigvals, eigvecs = eigh(LZ)  # 计算拉普拉斯矩阵的特征值和特征向量
    F = eigvecs[:, :k]  # 取前 k 个特征向量

    # 正交化 F，确保 F^T F = I
    # 这里使用SVD来保证F的列是正交的
    U, _, Vt = np.linalg.svd(F)
    F = U  # 使得 F 是正交的

    return F
