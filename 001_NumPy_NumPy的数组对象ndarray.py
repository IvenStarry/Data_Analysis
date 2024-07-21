import numpy as np
a = np.array([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]])
# ndarray五种属性
print(a.ndim)
print(a.shape)
print(a.size)
print(a.dtype)
print(a.itemsize)

# 可以由非同质对象构成(在numpy2.0.0版本中不支持 回退1.23.0版本才可以实现)
# 非同质ndarray对象无法发挥numpy优势 应避免书写
# x = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8]])
# print(x) 
# print(x.shape) # (2, )
# print(x.size) # 2
# # 元素为对象类型
# print(x.dtype) # ('0')
# print(x.itemsize) # 4