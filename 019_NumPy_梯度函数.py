import numpy as np
'''
gradient(f) 计算数组f元素的梯度 当f多维 返回每一个维度梯度 
一维数组：存在俩侧值 斜率=（右侧值-左侧值）/ 2
只存在一侧值 斜率=（本身-左侧值） 或者 （右侧值-本身）
二维数组：存在两个梯度值 横向一个 纵向一个 输出两个数组 第一个数组输出最外层维度梯度 第二个数组输出第第二层维度梯度
'''

a = np.random.randint(0, 20, (5))
print(a)
print(np.gradient(a))

b = np.random.randint(0, 50, (3, 5))
print(b)
print(np.gradient(b))