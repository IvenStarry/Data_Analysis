import numpy as np

# np.arange(staet, stop, step, dtype)不包含stop
x = np.arange(10)
print(x)
x = np.arange(10, dtype=float)
print(x)
x = np.arange(10, 20, 2)
print(x)

# np.linspace(start, end, num, endpoint=True, retstep=False, dtype) 
# num数组元素个数 4个元素有三个间隔 间隔为(10-1)/3=3
a = np.linspace(1, 10, 4)
print(a)
# 若设置endpoint为False 则不以end这个数结尾 但仍生成4个数 5个元素（包括10）有四个间隔 间隔变为(10-1)/4=2.25
b = np.linspace(1, 10, 4, endpoint=False)
print(b)
# 设置间距 retstep为True时，生成的数组显示间距
a = np.linspace(1, 10, 10, retstep=True)
print(a)
b = np.linspace(1, 10, 10).reshape([10,1])
print(b)

# np.logspace(s, s, num, endpoint, base=10.0, dtype) base log的对数
a = np.logspace(1.0, 2.0, 10)
print(a)


# # np.concatenate() 两个或多个数组合并成一个新的数组
# x = np.concatenate((a, b))
# print(x)


# # todo ndarray数组的变换
# a = np.ones((2, 3, 4), dtype=np.int32)
# print(a)
# # .reshape(shape) 返回新数组
# x = a.reshape(3, 8)
# print(x)
# # .resize(shape) 修改原数组
# a.resize(4, 6)
# print(a)
# # .swapaxes(ax1, ax2) 将数组的n个维度的2个维度调换 返回新数组 类似转置
# a = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
# print(a)
# b = a.swapaxes(0, 1)
# print(b)
# # .flatten() 数组降维，返回折叠后的一维数组，原数组不变
# c = a.flatten()
# print(c)
# # .astype(new_type) 转化数据类型 创建新数组
# print(a)
# x = a.astype(np.float64)
# print(x)
# # .tolist() ndarray数组向列表转换
# x = a.tolist()
# print(x)