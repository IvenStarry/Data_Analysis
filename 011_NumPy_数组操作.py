import numpy as np

# todo 修改数组形状
# reshape(a, newshape, order='C) 不改变数据的条件下修改形状
a = np.arange(8)
b = a.reshape(4,2)
print(a)
print(b)

# flat 数组元素迭代器
a = np.arange(9).reshape(3 ,3)
print('-----------------------------')
print(a)
for row in a:
    print(row)
for element in a.flat:
    print(element)

# numpy.ndarray.flatten(order='C) 数组降维，返回折叠后的一维数组，原数组不变
a = np.arange(8).reshape(2, 4)
print('-----------------------------')
print(a)
print(a.flatten())
print(a.flatten(order='F'))

# ravel(a, order='C) 展平的数组元素 返回数组视图 修改影响原数组 C按行 F按列 A原顺序 K元素在内存的出现顺序
a = np.arange(8).reshape(2, 4)
print('-----------------------------')
print(a)
print(a.ravel())
print(a.ravel(order='F'))

# todo 翻转数组
# transpose(a, axes) 对换数组维度
a = np.arange(12).reshape(3, 4)
print('-----------------------------')
print(a)
print(np.transpose(a))
print(a.T)

# rollaxis(a, axis, start) 函数向后滚动特定的轴到一个特定位置 axis 向后滚动的轴 axis滚动到start轴前面 其他轴相对位置不变
a = np.arange(8).reshape(2, 2, 2)
print('-----------------------------')
print(a)
print(np.where(a==6))
print(a[1, 1, 0])
# 将轴2滚动到轴0
b = np.rollaxis(a, 2, 0)
print(b)
print(np.where(b==6))
# 将轴2滚动到轴1
c = np.rollaxis(a, 2, 1)
print(c)
print(np.where(c==6))
'''
>>> a = np.ones((3,4,5,6))
>>> np.rollaxis(a, 3, 1).shape
(3, 6, 4, 5)
>>> np.rollaxis(a, 2).shape
(5, 3, 4, 6)
>>> np.rollaxis(a, 1, 4).shape
(3, 5, 6, 4)

三维数组array(轴0，轴1，轴2),将轴2滚动到轴0位置，其余轴顺序不变，即new_array(轴2，轴0，轴1)
元素[1,1,0]		[0,1,1]
'''

print('-----------------------------')
# .swapaxes(ax1, ax2) 将数组的n个维度的2个维度调换 返回新数组 类似转置
a = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
print(a)
b = a.swapaxes(0, 1)
print(b)

# todo 修改数组维度
print('-----------------------------')
# broadcast 模仿广播的对象 返回一个对象 该对象封装了将一个数组广播到另一个数组的结果
x = np.array([[1], [2], [3]])
y = np.array([4, 5, 6])
print(x.shape, y.shape)
b = np.broadcast(x, y) # 对y广播x 
r, c = b.iters # 自带迭代器属性
print(next(r), next(c))
print(next(r), next(c))
print(f'广播对象的形状{b.shape}')
b = np.broadcast(x, y)
c = np.empty(b.shape)
print(c.shape)
c.flat = [u + v for (u, v) in b]
print(c)
print(x + y)

print('-----------------------------')
# broadcast_to(a, shape, subok) 将数组广播到新形状 在原始数组返回只读视图 若新形状不符合广播规则，则error
a = np.arange(4).reshape(1, 4)
print(a)
print(np.broadcast_to(a, (4,4)))

print('-----------------------------')
# expand_dims(a, axis) 在指定位置插入新的轴扩展数组形状
a = np.array([[1, 2], [3, 4]])
print(a)
print(np.expand_dims(a, axis=0))
print(np.expand_dims(a, axis=0).shape, a.shape)
print(np.expand_dims(a, axis=1))
print(a.ndim, np.expand_dims(x, axis=1).ndim)
print(np.expand_dims(a, axis=1).shape, a.shape)

print('-----------------------------')
# squeeze(a, axis) 从给定数组的形状中删除一维的条目
a = np.arange(9).reshape(3, 1, 3)
print(a)
print(np.squeeze(a))
print(a.shape, np.squeeze(a).shape)

print('-----------------------------')
# todo 连接数组
# np.concatenate((a1, a2, ...), axis=0) 两个或多个数组合并成一个新的数组
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
x = np.concatenate((a, b))
print(x)
x = np.concatenate((a, b), 1)
print(x)

print('-----------------------------')
# stack(a, axis) 沿着新的轴堆叠数组序列
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print(np.stack((a, b), 0))
print(np.stack((a, b), 1))
print(np.stack((a, b), 2))

print('-----------------------------')
# hstack((a1, a2, ...)) 水平堆叠
print(np.hstack((a, b)))
print('-----------------------------')
# vstack((a1, a2, ...)) 垂直堆叠 
print(np.vstack((a, b)))

print('-----------------------------')
# todo 分割数组
# split(a, indices_or_sections, axis) 沿特定的轴将数组分割成子数组 
# indices_or_sections:若为整数，用该数平均切分；若为数组，为沿轴切分的位置（左开右闭）
# axis：沿着哪个方向切分 默认0 横向切分 为1纵向切分
a = np.arange(9)
print(a)
print(np.split(a, 3))
print(np.split(a, [4, 7]))
a = np.arange(16).reshape(4, 4)
print(a)
print(np.split(a, 2))
print(np.split(a, 2, 1))

print('-----------------------------')
# hsplit(a, num) 水平分割数组 num:要返回的相同形状的数组数量
a = np.arange(16).reshape(4, 4)
print(a)
print(np.hsplit(a, 2))

print('-----------------------------')
# vsplit 垂直分割数组
print(a)
print(np.vsplit(a, 2))

print('-----------------------------')
# todo 数组元素的添加与删除
# resize(a, shape) 修改原数组
a = np.arange(16).reshape(4, 4)
print(a)
print(np.resize(a, (2,8)))

print('-----------------------------')
# append(a, values, axis=None) None 横向加成返回一维数组
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a)
print(np.append(a, [7, 8, 9]))
print(np.append(a, [[7, 8, 9]], 0))
print(np.append(a, [[7, 8, 9], [10, 11, 12]], 1))

print('-----------------------------')
# insert(a, obj, values, axis) obj:在其之前插入值的索引 axis:未传入值则返回一维数组
a = np.array([[1, 2], [3, 4], [5, 6]])
print(a)
print(np.insert(a, 3, [11, 12]))
print(np.insert(a, 1, [11], axis=0)) # 轴0广播
print(np.insert(a, 1, 11, axis=1)) # 轴1广播

print('-----------------------------')
# delete(a, obj, axis) obj 删除的数字或数组
a = np.arange(12).reshape(3, 4)
print(a)
print(np.delete(a, 5))
print(np.delete(a, 1, axis=1))
print(np.delete(a, np.s_[::2]))

print('-----------------------------')
# unique(a, return_index, return_inverse, return_counts) 去除数组中的重复元素
# return_index True 返回新列表元素在旧列表中的位置（下标），并以列表形式存储
# return_inverse True 返回旧列表元素在新列表中的位置（下标），并以列表形式存储
# return_counts True 返回去重数组中的元素在原数组的出现次数
a = np.array([1, 1, 1, 2, 2, 3, 4, 5, 5, 5, 5, 6, 7, 7, 8, 9])
print(a)
print(np.unique(a, return_index=True))
print(np.unique(a, return_inverse=True))
print(np.unique(a, return_counts=True))


# .astype(new_type) 转化数据类型 创建新数组
print(a)
x = a.astype(np.float64)
print(x)

# .tolist() ndarray数组向列表转换
x = a.tolist()
print(type(x))