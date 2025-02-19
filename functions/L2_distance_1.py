import numpy as np

def L2_distance_1(a, b):
    """
    计算平方欧几里得距离
    ||A-B||^2 = ||A||^2 + ||B||^2 - 2*A'*B

    参数:
    a, b: 两个矩阵，每列是一个数��点
    返回:
    d: a和b的距离矩阵
    """
    if a.shape[0] == 1:
        a = np.vstack([a, np.zeros((1, a.shape[1]))])
        b = np.vstack([b, np.zeros((1, b.shape[1]))])

    aa = np.sum(a * a, axis=0)
    bb = np.sum(b * b, axis=0)
    ab = np.dot(a.T, b)

    d = np.tile(aa[:, np.newaxis], (1, bb.shape[0])) + np.tile(bb, (aa.shape[0], 1)) - 2 * ab

    d = np.real(d) / 2
    d = np.maximum(d, 0)

    # 强制对角线为0
    # if df == 1:
    #     d = d * (1 - np.eye(d.shape[0]))

    return d