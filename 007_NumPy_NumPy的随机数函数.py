import numpy as np
# rand() 随机数数组 浮点数 符合均匀分布 [0, 1)
a = np.random.rand(3, 4, 5)
print(a)

# randn() 标准正态分布
a = np.random.randn(3, 4, 5)
print(a)

# randint(low, high, shape) 随机整数数组 [low, high)
# seed() 使用同样的seed种子 生成的随机数数组相同
np.random.seed(10)
a = np.random.randint(100, 200, (3, 4))
print(a)
np.random.seed(10)
b = np.random.randint(100, 200, (3, 4))
print(b)

# shuffle 数组的第一轴进行随即变换 改变原数组
np.random.shuffle(a)
print(a)
np.random.shuffle(a)
print(a)

# permutation() 数组的第一轴进行随机变换 不改变原数组 生成新数组
b = np.random.permutation(a)
print(b)

# choice(a, size, replace, p) 从一维数组中以概率p抽取元素 形成size形状新数组 replace是否重用新元素 默认True值
a = np.random.randint(10, 20, (8,))
print(a)
print(np.random.choice(a, (3,2)))
print(np.random.choice(a, (3,2), replace=False))
print(np.random.choice(a, (3,2), p=a/np.sum(a)))

# uniform(low, high, size) 均匀分布数组
print(np.random.uniform(0, 10, (3, 4)))
# normal(loc, scale, size) 正态分布数组 loc均值 scale标准差
print(np.random.normal(10, 5, (3, 4)))
# poisson(lam, size)       柏松分布数组 lam随机事件发生率 
print(np.random.poisson(0.5, (3, 4)))