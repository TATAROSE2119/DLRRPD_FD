import numpy as np

def EProjSimplex_new(v, k=1):
    """
    解决以下问题:
    min  1/2 || x - v||^2
    s.t. x >= 0, 1'x = k

    参数:
    v: 输入向量
    k: 约束条件，默认为1

    返回:
    x: 投影后的向量
    ft: 迭代次数
    """
    ft = 1
    n = len(v)

    v0 = v - np.mean(v) + k / n
    vmin = np.min(v0)
    if vmin < 0:
        f = 1
        lambda_m = 0
        while abs(f) > 1e-10:
            v1 = v0 - lambda_m
            posidx = v1 > 0
            npos = np.sum(posidx)
            g = -npos
            f = np.sum(v1[posidx]) - k
            lambda_m = lambda_m - f / g
            ft += 1
            if ft > 100:
                x = np.maximum(v1, 0)
                break
        else:
            x = np.maximum(v1, 0)
    else:
        x = v0

    return x