import numpy as np
# 迭代器np.nditer 可以完成对数组元素的访问

a = np.arange(6).reshape(2, 3)
print(a)
print("迭代输出数组：")
for x in np.nditer(a):
    print(x, end=' ')
print('\n---------------------')
for x in np.nditer(a.T):
    print(x, end=' ')
print('\n---------------------')
for x in np.nditer(a.T.copy(order='C')):
    print(x, end=' ')
# a和a.T的遍历顺序一样，是因为选择的顺序和数组a的内存布局一样（行序优先C） 但a.T.copy(order='C')的遍历结果不一样，因为指定了a.T的行序优先

# 在copy中控制遍历排序
print('\n---------------------')
print("以C风格顺序排序")
b = a.T.copy(order='C')
for x in np.nditer(b):
    print(x, end=' ')
print('\n---------------------')
print("以F风格顺序排序")
b = a.T.copy(order='F')
for x in np.nditer(b):
    print(x, end=' ')

# 在nditer中控制遍历排序
print('\n---------------------')
print("以C风格顺序排序")
for x in np.nditer(a.T, order='C'):
    print(x, end=' ')
print('\n---------------------')
print("以F风格顺序排序")
for x in np.nditer(a.T, order='F'):
    print(x, end=' ')

# nditer对象默认将遍历的数组视为只读对象，设置参数op_flags可以在遍历同时，对数组进行修改，指定readwrite或writeonly模式
print('修改之前的数组：')
print(a)
for x in np.nditer(a, op_flags=['readwrite']):
    # * 使用x = 2 + x是无法完成对数组a的修改操作的。因为直接赋值操作将会创建一个新的数组，而不会修改原始的数组 a。
    # * 要修改原始数组 a 中的值，需要使用 x[...] = 2 + x 这种形式的赋值语句，以确保对a进行原地修改。
    x[...] = 2 + x
print('修改之后的数组：')
print(a)

# flags参数： c_index 跟随C顺序的索引 f_index 跟随Fortan顺序的索引 multi_index 每次迭代可以跟踪一种索引类型 external_loop 给出的值是具有多个值的一维数组而不是零维数组
for x in np.nditer(a, flags = ["external_loop"], order = 'F'):
    print(x, end=' ')

# 广播迭代 a(3,4) b(1,4)
a = np.arange(0, 60, 5).reshape(3, 4)
b = np.array([1, 2, 3, 4], dtype=int)
print('\on')
print(a)
print(b)
for x, y in np.nditer([a, b]):
    print(f"{x},{y}", end=' ')