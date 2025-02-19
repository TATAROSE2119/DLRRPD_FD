import numpy as np

from functions.eig1 import eig1
from functions.L2_distance_1 import L2_distance_1
from functions.EProjSimplex_new import EProjSimplex_new
from tqdm import tqdm

def DLRRPD(X, F_ini, Z_ini, c, lambda1, lambda2, lambda3, max_iter):
    """
    该代码由Zhiqiang Fu编写。
    """
    n, m = X.shape if X.shape[0] > X.shape[1] else X.shape[::-1]
    # 初始化
    miu = 0.01
    rho = 1.2
    max_miu = 1e8
    tol = 1e-5
    tol2 = 1e-2
    zr = 1e-9
    C1 = np.zeros((m, n))
    C2 = np.zeros((n, n))
    E = np.zeros((m, n))
    P = np.zeros((m, m))

    objective_values=[]

    for iter in tqdm(range(max_iter),desc="DLRRPD Progress"):
        if iter == 0:
            Z = Z_ini
            F = F_ini
            S = Z_ini
            del Z_ini, F_ini

        S_old = S.copy()
        P_old = P.copy()
        Z_old = Z.copy()
        E_old = E.copy()

        # 更新Z
        PX = P @ X
        Z = np.linalg.pinv((lambda2 + miu) * np.eye(n) + miu * (PX.T @ PX)) @ (miu * (PX.T @ (X - E + C1 / miu) + S - C2 / miu))
        Z = Z - np.diag(np.diag(Z))

        # 更新S
        distX = L2_distance_1(PX, PX)
        distF = L2_distance_1(F.T, F.T)
        dist = distX + lambda1 * distF
        S = Z + (C2 - dist) / miu
        S = S - np.diag(np.diag(S)) # 对角线元素为0
        for ic in range(n):
            idx = list(range(n))
            idx.remove(ic)
            S[ic, idx] = EProjSimplex_new(S[ic, idx])

        # 更新F
        LS = (S + S.T) / 2
        L = np.diag(np.sum(LS, axis=1)) - LS
        F, _, ev = eig1(L, c, 0)

        # 更新P
        Wz = (S + S.T) / 2
        Dz = np.diag(np.sum(Wz, axis=1))
        Lz = Dz - Wz
        M = X @ Lz @ X.T
        XZ = X @ Z
        P = -miu * (E - X - C1 / miu) @ XZ.T @ np.linalg.pinv(np.eye(m) + 4 * M + miu * XZ @ XZ.T)

        # 更新E
        temp1 = X - P @ X @ Z + C1 / miu
        temp2 = lambda3 / miu
        E = np.maximum(0, temp1 - temp2) + np.minimum(0, temp1 + temp2)

        # 更新C1, C2, miu
        L1 = X - P @ X @ Z - E
        L2 = Z - S
        C1 = C1 + miu * L1
        C2 = C2 + miu * L2
        LL1 = np.linalg.norm(Z - Z_old, 'fro')
        LL2 = np.linalg.norm(S - S_old, 'fro')
        LL3 = np.linalg.norm(P - P_old, 'fro')
        LL4 = np.linalg.norm(E - E_old, 'fro')
        SLSL = max(max(max(LL1, LL2), LL3), LL4) / np.linalg.norm(X, 'fro')
        miu = min(rho * miu, max_miu)
        # 计算目标函数值------------------------------------
        # 计算 Frobenius 范数
        def frobenius_norm(M):
            return np.linalg.norm(M, 'fro') ** 2

        # 计算目标函数的每一项
        term_1 = 2 * np.trace(P @ X @ L @ Z @ X.T @ P.T)
        term_2 = lambda1 / 2 * (frobenius_norm(Z) + frobenius_norm(P))
        term_3 = lambda2 * np.linalg.norm(E, 1)  # L1 范数
        term_4 = (2 * lambda3 * np.trace(F.T @ L @ Z @ F))/ frobenius_norm(X)

        # 计算目标函数值
        objective_value = term_1 + term_2 + term_3 + term_4
        #将目标函数值存入list
        objective_values.append(objective_value)
        #---------------------------------------------------

        # 检查收敛性
        leq1 = max(np.max(np.abs(L1)), np.max(np.abs(L2)))
        stopC = leq1
        if stopC < tol:
            print(f'在迭代 {iter} 时收敛')
            break

    return Z, S, P, F, E ,stopC,objective_values


# 需要定义辅助函数（L2_distance_1, EProjSimplex_new, eig1）。