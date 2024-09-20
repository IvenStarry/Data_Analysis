import numpy as np

# 整数数组索引 使用一个数组访问另一个数组元素
x = np.array([[1, 2], [3, 4], [5, 6]])
y = x[[0, 1, 2], [0, 1, 0]] # 第一个维度的索引[0, 1, 2] 第二个维度的索引[0, 1, 0]
print(y)

x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
print(f'数组是:\n{x}')
rows = [[0, 0], [3, 3]]
cols = [[0, 2], [0, 2]]
print(f'四个角的元素为:\n{x[rows, cols]}')

a = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
b = a[1:3, 1:3]
c = a[1:3, [1,2]]
d = a[...,1:]
print(b)
print(c)
print(d)

# 布尔索引 通过布尔运算获取符合指定条件的元素的数组
x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
print(x[x > 5])
a = np.array([np.nan, 1, 2, np.nan, 3, 4, 5]) # nan 非数字元素
print(a[~np.isnan(a)]) # isnan检测数组的非数字元素 ~取补运算符
a = np.array([1, 2+6j, 2, 5J, 5])
print(a[np.iscomplex(a)]) # iscomplex检测数组的复数元素

# 花式索引 根据索引数组的值作为目标数组的某个轴的下标来取值
x = np.arange(9)
print(x)
y = x[[0, 6]]
print(y)
print(y[0])
print(y[1])

# 二维数组
x = np.arange(32).reshape((8,4))
print(x)
print(x[[4, 2, 1, 7]])
print(x[[-4, -2, -1, -7]])
"""
np.ix_ 输入两个数组，产生笛卡尔积的映射关系
e.g. A=(a,b) B=(0,1,2)
A * B = {(a,0), (a,1), (a,2), (b,0), (b,1), (b,2)}
B * A = {(0,a), (1,a), (2,a), (0,b), (1,b), (2,b)}
"""
x = np.arange(32).reshape((8,-1))
print(x[np.ix_([1,5,7,2],[0,3,1,2])])