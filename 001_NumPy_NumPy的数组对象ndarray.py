import numpy as np
# x = np.array(list/tuple, dtype='int32', ndmin) 可以指定数据类型
# 从列表
x = np.array([0, 1, 2, 3], dtype=np.float64)
print(x)
x = np.array([1, 2, 3], dtype=complex)
print(x)
# 从元组
x = np.array((0, 1, 2, 3))
print(x)
# 从列表和元组混合类型创建
x = np.array(([0, 1], [1, 2], [2, 3], [3, 4]))
print(x)
# ndmin 指定生成数组的最小维度
x = np.array([1, 2, 3, 4, 5], ndmin=2)
print(x)