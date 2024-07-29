import pandas as pd
import numpy as np

# 创建Series
'''
pandas.Series(data=None, index=None, dtype=None, name=None, copy=False, fastpath=False)

data：Series 的数据部分，可以是列表、数组、字典、标量值等。如果不提供此参数，则创建一个空的 Series。
index：Series 的索引部分，用于对数据进行标记。可以是列表、数组、索引对象等。如果不提供此参数，则创建一个默认的整数索引。
dtype：指定 Series 的数据类型。可以是 NumPy 的数据类型，例如 np.int64、np.float64 等。如果不提供此参数，则根据数据自动推断数据类型。
name：Series 的名称，用于标识 Series 对象。如果提供了此参数，则创建的 Series 对象将具有指定的名称。
copy：是否复制数据。默认为 False，表示不复制数据。如果设置为 True，则复制输入的数据。
fastpath：是否启用快速路径。默认为 False。启用快速路径可能会在某些情况下提高性能。
'''
a = [1, 2, 3]
myvar = pd.Series(a)
print(myvar) # 索引在左 数据在右 数据类型在下
print(myvar[1]) # 指定索引

# 指定索引值
a = ['Iven', 'Rosenn', 'Starry']
myvar = pd.Series(a, index=['x', 'y', 'z'])
print(myvar['y'])

# 也可以用key/value对象，类似字典来创建Series
sites = {1:'Iven', 2:'Rosenn', 3:'Starry'}
myvar = pd.Series(sites)
print(myvar)
# 可以看出key变成了索引值，可以通过key取出数据
print(myvar[1])

# 如果只需要字典中的一部分数据，只需要指定需要数据的索引即可
myvar = pd.Series(sites, index=[1, 2])
print(myvar)

# 设置Series名称参数
myvar = pd.Series(sites, index=[1, 2], name='Iven_Series_test')
print(myvar)

# 使用列表、字典或数组创建一个默认索引的Series
s = pd.Series([1, 2, 3, 4])
s = pd.Series(np.array([1, 2, 3, 4]))
s = pd.Series({'a':1, 'b':2, 'c':3, 'd':4})

# todo 基本操作
# 指定索引创建Series
s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])

# 获取值
value = s[2] # 不良好的书写习惯 尽量使用数据标签获取值
print(value)
print(s['a'])

# 获取多个值
subset = s[1:4]
print(subset)

# 使用自定义索引
value = s['b']
print(value)

# 索引与值对应的关系
for index, value in s.items():
    print(f'Index: {index}, value:{value}')

# 使用切片语法来访问Series的一部分
print(s['a':'c'])
print(s[:3])

# 为特定的索引标签赋值
s['a'] = 10

# 通过赋值给新的索引标签来添加元素
s['e'] = 5

# 使用 del 删除指定索引标签的元素
del s['a']

# 使用 drop 方法删除一个或多个索引标签，返回一个新的Series
s_dropped = s.drop(['b'])
print(s_dropped)

# todo 基本运算
# 算术运算
result = s_dropped * 2 # 所有元素乘2
print(result)

# 过滤
filtered_series = s_dropped[s_dropped > 3]
print(filtered_series)

# 数学函数
result = np.sqrt(s_dropped)
print(result)

# todo 计算统计数据 使用Series的方法来计算描述性统计
print(s_dropped.sum()) # 计算Series的总和
print(s_dropped.mean()) # 平均值
print(s_dropped.max()) # 最大值
print(s_dropped.min()) # 最小值
print(s_dropped.std()) # 标准差

# todo 属性和方法
# 获取索引
index = s.index
print(index)

# 获取值数组
values = s.values
print(values)

# 获取描述统计信息
stats = s.describe()
print(stats)

# 获取最大值最小值索引
max_index = s.idxmax()
print(max_index)
min_index = s.idxmin()
print(min_index)

# 其他属性和方法
print(s.dtype) # 数据类型
print(s.shape) # 形状
print(s.size)  # 元素个数
print(s.head())# 前几个元素，默认前五个
print(s.tail())# 后几个元素，默认后五个
print(s.sum()) # 求和
print(s.mean())# 平均值
print(s.max()) # 最大值
print(s.min()) # 最小值
print(s.std()) # 标准差

# 布尔表达式 根据条件过滤Series
print(s > 3)

# 转换数据类型 astype 将Series转换为另一种数据类型
s_astype = s.astype('float64')
print(s.dtype)
print(s_astype.dtype)