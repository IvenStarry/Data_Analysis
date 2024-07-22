'''
索引：获取数组中特定位置的元素
切片：获取数组元素子集的过程
'''
import numpy as np
# 一维数组
a = np.array([11, 22, 33, 44, 55])
print(a[2])
print(a[1:4:2]) # 同python列表 start end(not include) step

# 多维数组
a = np.arange(24).reshape((2, 3, 4))
print(a)
print(a[1, 2, 3])
print(a[0, 1, 2])
print(a[-1, -2, -3])
print(a[:, 1, -3]) # 第一个维度全选 第二个维度取索引为1的元素 第三个维度选索引为-3的元素
print(a[:, 1:3, :])
print(a[:, :, ::2])