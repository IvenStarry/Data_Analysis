import numpy as np
a = np.array([[0, 1, 2, 3, 4], [9, 8, 7, 6, 5]])

"""
ndarray对象属性：
1. .ndim 轴的数量或维度的数量
2. .shape 对象的尺度，对于矩阵 n行m列
3. .size 对象元素的个数 相当于n*m
4. .dtype 元素类型
5. .itemsize 对象的每个元素的大小，以字节为单位
6. .flags 返回ndarray对象的内存信息 ：包含了
    C_CONTIGUOUS : True      数据在一个单一C风格的连续段中
    F_CONTIGUOUS : False     数据是在一个单一的Fortan风格的连续段中
    OWNDATA : True           数组拥有它所使用的内存或从另外一个对象中借用它  
    WRITEABLE : True         数据区域可以被写入 若False则只可读  
    ALIGNED : True           数据和元素都适当对齐在硬件上
    WRITEBACKIFCOPY : False  这个数组是其他数组的一个副本。当这个数组被释放时，原数组将更新
"""
print(a.ndim)
print(a.shape)
print(a.size)
print(a.dtype)
print(a.itemsize)
print(a.flags)

# 可以由非同质对象构成(在numpy2.0.0版本中不支持 回退1.23.0版本才可以实现)
# 非同质ndarray对象无法发挥numpy优势 应避免书写
# x = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8]])
# print(x) 
# print(x.shape) # (2, )
# print(x.size) # 2
# # 元素为对象类型
# print(x.dtype) # ('0')
# print(x.itemsize) # 4