import numpy as np

def eig1(A, c=None, isMax=True, isSym=True):
    """
    计算矩阵A的前c个特征值和特征向量。

    参数:
    A: 输入矩阵
    c: 需要的特征值和特征向量的数量
    isMax: 是否按特征值从大到小排序
    isSym: 是否将矩阵视为对称矩阵

    返回:
    eigvec: 前c个特征向量
    eigval: 前c个特征值
    eigval_full: 所有特征值
    """
    if c is None:
        c = A.shape[0]
    elif c > A.shape[0]:
        c = A.shape[0]

    if isSym:
        A = np.maximum(A, A.T)

    # 计算特征值和特征向量
    eigval_full, eigvec = np.linalg.eig(A)

    # 按特征值排序
    if isMax:
        idx = np.argsort(eigval_full)[::-1]  # 从大到小排序
    else:
        idx = np.argsort(eigval_full)  # 从小到大排序

    eigval_full = eigval_full[idx]
    eigvec = eigvec[:, idx]

    eigval = eigval_full[:c]
    eigvec = eigvec[:, :c]

    return eigvec, eigval, eigval_full