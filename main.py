import numpy as np
from functions import DLRRPD
from functions.initialize_F import initialize_F
#from functions.knnG import compute_knn_graph
from functions.knnG2 import compute_knn_graph
from functions.DLRRPD import DLRRPD
import scipy.io as sio
# 读取数据
# 读取.mat文件
mat_contents = sio.loadmat('other_data/auto_uni.mat')
# 假设数据存储在键 'data' 下
train_data = mat_contents['X']

#train_data=np.loadtxt('TE_data/train_data/d00.dat')

if train_data.shape[0] < train_data.shape[1]:
    train_data = train_data.T

train_data = train_data.T
n_samples, n_features = train_data.shape if train_data.shape[0] > train_data.shape[1] else train_data.shape[::-1]

ini_Z = compute_knn_graph(train_data, 10)
ini_F = initialize_F(ini_Z, 10)

Z, S, P, F, E ,stopC,objective_values=DLRRPD(train_data, ini_F, ini_Z, 10, 100, 10, 1, 50)
print(stopC)
#绘制objective_values
import matplotlib.pyplot as plt
plt.plot(objective_values)
plt.xlabel('Iteration')
plt.ylabel('Objective Value')
plt.title('DLRRPD Objective Value')
plt.show()
