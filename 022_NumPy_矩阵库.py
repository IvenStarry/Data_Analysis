import numpy.matlib
import numpy as np
# numpy包含矩阵库numpy.matlib，模块返回一个矩阵，而非ndarray对象

# 转置矩阵
a = np.arange(12).reshape(3, 4)
print(a)
print(a.T)
print(type(a), type(a.T))

# matlib.empty 返回一个新的矩阵
a = numpy.matlib.empty((2, 2))
print(a)
print(type(a))

# matlib.zeros 返回全0矩阵
a = numpy.matlib.zeros((2, 2))
print(a)

# matlib.ones 创建全1矩阵
a = numpy.matlib.ones((2, 2))
print(a)

# matlib.eye(row, col, k, dtype) 创建对角线为1其他位置为0的矩阵  k:对角线的索引
a = numpy.matlib.eye(3, 3, k=0, dtype=float)
print(a)
a = numpy.matlib.eye(3, 3, k=1, dtype=float)
print(a)

# matlib.identity() 返回给定大小的单位矩阵
a = numpy.matlib.identity(3, dtype=float)
print(a)

# matlib.rand 随机数矩阵
a = numpy.matlib.rand(3, 3)
print(a)

# 矩阵总是二维的，而ndarray是一个n维数组，二者可以互换
a = np.matrix('1, 2; 3, 4')
print(a, type(a))
b = np.asarray(a) # asarray：矩阵转ndarray
print(b, type(b))
c = np.asmatrix(b) # asmatrix：adarray转矩阵
print(c, type(c))