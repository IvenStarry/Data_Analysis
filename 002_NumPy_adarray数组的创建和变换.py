import numpy as np

# todo 1.从python的列表、元组等类型创建adarray数组
# x = np.array(list/tuple, dtype='int32') 可以指定数据类型
# 从列表
x = np.array([0, 1, 2, 3], dtype=np.float64)
print(x)
# 从元组
x = np.array((0, 1, 2, 3))
print(x)
# 从列表和元组混合类型创建
x = np.array(([0, 1], [1, 2], [2, 3], [3, 4]))
print(x)

# todo 2.使用NumPy中函数创建adarray数组
# np.arange(n) 生成从0到n-1的adarray数组 返回整数型数据
x = np.arange(10)
print(x)
# np.ones(shape) 生成和shape大小一致的全1矩阵，shape是元组类型 数据类型默认浮点型
x = np.ones((2, 5))
print(x)
x = np.ones((2, 3, 4)) # 从外到内 最外层两个元素 每个元素三个维度 每个维度四个元素
print(x)
# np.zeros(shape) 生成和shape大小一致的全0矩阵，shape是元组类型 数据类型默认浮点型(可以指定数据类型)
x = np.zeros((3, 4), dtype=np.int_)
print(x)
# np.full(shape, val) 生成和shape大小一致的全val矩阵
x = np.full((3, 4), 10)
print(x)
# np.eye(n) 生成n*n的单位矩阵 对角线为1 其余为0 数据类型默认浮点型
x = np.eye(5)
print(x)

a = [[[1, 2], [2, 3]], [[3, 4], [4, 5]], [[5, 6], [6, 7]]]
# np.ones_like(a) 根据数组a的形状生成全1数组
x = np.ones_like(a)
print(x)
# np.zeros_like(a) 根据数组a的形状生成全0数组
x = np.zeros_like(a)
print(x)
# np.full_like(a, val) 根据数组a的形状生成全val数组
x = np.full_like(a, 5)
print(x)

# todo 3.使用NumPy中其他函数创建adarray数组
# np.linspace(start, end, num) num数组元素个数 4个元素有三个间隔 间隔为(10-1)/3=3
a = np.linspace(1, 10, 4)
print(a)
# 若设置endpoint则不以end这个数结尾 但仍生成4个数 5个元素（包括10）有四个间隔 间隔变为(10-1)/4=2.25
b = np.linspace(1, 10, 4, endpoint=False)
print(b)
# np,concatenate() 两个或多个数组合并成一个新的数组
x = np.concatenate((a, b))
print(x)

# todo ndarray数组的变换
a = np.ones((2, 3, 4), dtype=np.int32)
print(a)
# .reshape(shape) 返回新数组
x = a.reshape(3, 8)
print(x)
# .resize(shape) 修改原数组
a.resize(4, 6)
print(a)
# .swapaxes(ax1, ax2) 将数组的n个维度的2个维度调换 返回新数组 类似转置
a = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
print(a)
b = a.swapaxes(0, 1)
print(b)
# .flatten() 数组降维，返回折叠后的一维数组，原数组不变
c = a.flatten()
print(c)
# .astype(new_type) 转化数据类型 创建新数组
print(a)
x = a.astype(np.float64)
print(x)
# .tolist() ndarray数组向列表转换
x = a.tolist()
print(x)