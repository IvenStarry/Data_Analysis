import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])
c = a * b
print(c)

# 广播Broadcast 对不同形状的数组进行数值计算的方式
a = np.array([[0, 0, 0], [10, 10, 10], [20, 20, 20], [30, 30, 30]])
b = np.array([0, 1, 2])
print(b.shape)
print(a + b) # 将b延伸至与a维度大小相同

# np.tile(a, reps) reps：对应的英文单词为repeats，list表示，reps表示对A的各个axis进行重复的次数
bb = np.tile(b, (4, 1))
print(bb)
print(a + bb)
"""
广播触发机制 两个数组a b 
两个矩阵在一个维度上数据宽度相同 但在另一个维度上数据宽度不同
并且形状小的矩阵 在数据宽度不同的的这一维度只有一个元素
e.g.a.shape=(4,3)而b.shape=(1,3)，两个矩阵axis=1的数据宽度是相同的，但是axis=0的数据宽度不一样，
并且b.shape[0]=1，这就是广播机制的触发条件，numpy会把b沿axis=0的方向复制4份，即形状变成(4, 3)，与a的一致，接下来就是对应位相加即可
"""