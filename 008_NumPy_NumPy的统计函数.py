import numpy as np

a = np.arange(15).reshape(3, 5)
print(a)

# sum(a, axis=None) 计算数组a的axis轴所有元素之和 默认全部轴
print(np.sum(a))

# mean(a, axis=None) 计算数组a的axis轴所有元素期望 默认全部轴
print(np.mean(a, axis=0))
print(np.mean(a, axis=1))

# average(a, axis=None, weights=None) 计算数组a的axis轴所有元素加权平均值 默认全部轴
print(np.average(a, axis=0, weights=[10, 5, 1]))

# std(a, axis=None) 计算数组a的axis轴所有元素标准差 默认全部轴
print(np.std(a))

# var(a, axis=None) 计算数组a的axis轴所有元素方差 默认全部轴
print(np.var(a))

a = np.arange(15, 0, -1).reshape(3, 5)

# min(a) max(a) 最小值最大值
print(np.min(a), np.max(a))

# argmin(a) argmax(a) a中元素最小值、最大值的降到一维后下标
print(np.argmin(a), np.argmax(a))

# unravel_index(index, shape) 根据shape将一维下标index转多维下标
print(np.unravel_index(np.argmin(a), a.shape))

# ptp(a) a中元素最大值和最小值之差
print(np.ptp(a))

# median(a) 元素中位数
print(np.median(a))