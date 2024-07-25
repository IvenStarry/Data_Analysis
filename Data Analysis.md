# Python数据分析与展示
Github：https://github.com/IvenStarry  
学习视频网站：中国大学MOOC北京理工大学嵩天教授 https://www.icourse163.org/course/BIT-1001870002?tid=1472922453  
菜鸟网 https://www.runoob.com/

## NumPy
### 数据的维度
维度是一组数据的组织形式
一维数据有对等关系的有序或无序数据构成，采用线性方式组织

列表和数组区别
列表：数据类型可以不同
数组：数据类型相同

二维数据由多个一维数据构成，是一维数据的组合形式（表格）
多维数据由一维或二维数据在新维度上扩展得来
高维数据仅利用最近本的二元关系展示数据间的复杂结构（字典）

数据维度的Python表示
一维数据：列表和集合
二维数据：列表
多维数据：列表
高维数据：字典或数据表示格式（JSON、XML、YAML）

### NumPy的数组对象ndarray
NumPy是一个开源的Python科学计算基础库
有一个强大的N维数组对象ndarray
广播功能函数
整合C/C++/Fortran代码的工具
线性代数、傅里叶变换、随机数生成等功能
是SciPy、Pandas等数据处理或科学计算库的基础

N维数组对象ndarray
数组对象可以去掉元素间运算所需的循环，使一维向量更像单个数据
设置专门的数组对象，提升运算速度（底层代码由C语言编写）
采用相同的数据类型，助于节省运算和存储空间
ndarray是一个多维数组对象，由两部分构成：实际的数据，描述这些数据的元数据（数据维度、数据类型）
一般要求所有元素类型相同，数组下标从0开始

np.array()生成一个ndarray数组 输出成[]形式，元素用空格分割
轴(axis):保存数据的维度 秩:轴的数量

![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407211328315.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407211329907.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407211329850.png)
python仅支持整数、浮点数和复数3种类型
科学计算设计数据较多，对存储和性能有较高要求，对元素类型精确定义有助于NumPy合理使用存储空间并优化性能，也有助于程序员对程序规模合理评估
```python
import numpy as np
# x = np.array(list/tuple, dtype='int32', ndmin) 可以指定数据类型
# 从列表
x = np.array([0, 1, 2, 3], dtype=np.float64)
print(x)
x = np.array([1, 2, 3], dtype=complex)
print(x)
# 从元组
x = np.array((0, 1, 2, 3))
print(x)
# 从列表和元组混合类型创建
x = np.array(([0, 1], [1, 2], [2, 3], [3, 4]))
print(x)
# ndmin 指定生成数组的最小维度
x = np.array([1, 2, 3, 4, 5], ndmin=2)
print(x)
```

### 数据类型
```python
import numpy as np
# bool_ int_ uint8 float_ complex_

# todo dtype(object, align, copy) 转换为的数据类型对象 True填充字段使其类似C的结构体 复制dtype对象，若为false则是对内置数据类型对象的引用
dt = np.dtype(np.int32)
print(dt)
# int8 int16 int32 int64 四种数据类型可以使用字符串'i1' 'i2' 'i4' 'i8'代替
dt = np.dtype('i4')
print(dt)
# 字节顺序标注  <意味着小端法（低位字节存储在低位地址） >意味着大端法（高位字节存放在低位地址）
dt = np.dtype('<i4')
print(dt)
# 创建结构化数据类型
dt = np.dtype([('age', np.int8)])
print(dt)
# 数据类型应用于ndarray对象
a = np.array([(10, ), (20, ), (30, )], dtype=dt)
print(a)
# 类型字段名用于存取实际的age列
print(a['age'])
# 定义一个结构化数据类型，将这个dtype应用于ndarray对象
student = np.dtype([('name', 'S20'), ('age', 'i1'), ('marks', 'f4')])
print(student)
a = np.array([('abc', 21, 50), ('xyz', 18, 75)], dtype=student)
print(a)

"""
内建类型符号
b bool
i int
u uint
f float
c 复数浮点型
m timedelta 时间间隔
M datetime 日期时间
O python对象
S, a (byte)字符串
U unicode 统一码 编码格式 数字到文字的映射
V 原始数据(void)
"""
```

### 数据属性
```python
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
```

### 创建数组
```python
import numpy as np

# empty(shape, dtype=float, order='C') 生成未初始化数组 order 可选"C"或"F"代表行优先或列有限，在计算机内存中的存储元素的顺序
x = np.empty([3, 2], dtype=int)
print(x) # 数组元素随机值

# np.arange(n) 生成从0到n-1的adarray数组 返回整数型数据
x = np.arange(10)
print(x)

# np.ones(shape) 生成和shape大小一致的全1矩阵，shape是元组类型 数据类型默认浮点型
x = np.ones((2, 5))
print(x)
x = np.ones((2, 3, 4)) # 从外到内 最外层两个元素 每个元素三个维度 每个维度四个元素
print(x)

# np.zeros(shape) 生成和shape大小一致的全0矩阵，shape是元组类型 数据类型默认浮点型(可以指定数据类型)
x = np.zeros((3, 4), dtype=np.int_)
print(x)

# np.full(shape, val) 生成和shape大小一致的全val矩阵
x = np.full((3, 4), 10)
print(x)

# np.eye(n) 生成n*n的单位矩阵 对角线为1 其余为0 数据类型默认浮点型
x = np.eye(5)
print(x)

a = [[[1, 2], [2, 3]], [[3, 4], [4, 5]], [[5, 6], [6, 7]]]
# np.ones_like(a) 根据数组a的形状生成全1数组 subok 为True时，使用object的内部数据类型； 为False时，使用数组的数据类型
#创建矩阵
a=np.asmatrix([1,2,3,4])
#输出为矩阵类型
print(type(a))
#既要赋值一份副本，又要保持原类型
at=np.array(a,subok=True)
af=np.array(a) #默认为False
print('at.subok为True:',type(at))
print('af.subok为False:',type(af))
print(id(af),id(a))

# np.zeros_like(a,order="K") 根据数组a的形状生成全0数组 order默认K保留输入数组的存储顺序 可选C F
x = np.zeros_like(a)
print(x)

# np.full_like(a, val) 根据数组a的形状生成全val数组
x = np.full_like(a, 5)
print(x)
```

### 从已有的数组创建数组
```python
import numpy as np

# asarray(a, dtype, order) 类似array 参数比array少俩
x = [1, 2, 3]
a = np.asarray(x)
print(a)

# 元组转换ndarray
x = (1, 2, 3)
a = np.asarray(x)
print(a)

# 元组列表转换ndarray 高版本无法生成非同质数组
# // x = [(1, 2 , 3), (4, 5)]
# // a = np.asarray(x)
# // print(a)

# frombuffer(buffer, dtype, count=-1, offset) 实现动态数组 接受buffer输入参数 以流的形式读取转化成ndarray对象 
# offset读取起始位置 默认0   b" "前缀表示：后面字符串是bytes 类型
s = b'Hello World!'
a = np.frombuffer(s, dtype='S1')
print(a)

# fromiter(iterable, dtype, count=-1) 从迭代对象中建立ndarray对象，返回一维数组
list = range(5)
it = iter(list)
x = np.fromiter(it, dtype=float)
print(x)
```

### 从数值范围创建数组
```python
import numpy as np

# np.arange(staet, stop, step, dtype)不包含stop
x = np.arange(10)
print(x)
x = np.arange(10, dtype=float)
print(x)
x = np.arange(10, 20, 2)
print(x)

# np.linspace(start, end, num, endpoint=True, retstep=False, dtype) 
# num数组元素个数 4个元素有三个间隔 间隔为(10-1)/3=3
a = np.linspace(1, 10, 4)
print(a)
# 若设置endpoint为False 则不以end这个数结尾 但仍生成4个数 5个元素（包括10）有四个间隔 间隔变为(10-1)/4=2.25
b = np.linspace(1, 10, 4, endpoint=False)
print(b)
# 设置间距 retstep为True时，生成的数组显示间距
a = np.linspace(1, 10, 10, retstep=True)
print(a)
b = np.linspace(1, 10, 10).reshape([10,1])
print(b)

# np.logspace(s, s, num, endpoint, base=10.0, dtype) base log的对数
a = np.logspace(1.0, 2.0, 10)
print(a)
```

### 切片和索引
```python
'''
索引：获取数组中特定位置的元素
切片：获取数组元素子集的过程
'''
import numpy as np

# 一维数组
a = np.array([11, 22, 33, 44, 55])
print(a[2])
s = slice(1, 4, 2) # 索引2 到索引4停止
print(a[s])
print(a[1:4:2]) # 同python列表 start end(not include) step

# 多维数组
a = np.arange(24).reshape((2, 3, 4))
print(a)
print(a[1, 2, 3])
print(a[0, 1, 2])
print(a[-1, -2, -3])
print(a[:, 1, -3]) # 第一个维度全选 第二个维度取索引为1的元素 第三个维度选索引为-3的元素
print(a[:, 1:3, :])
print(a[:, :, ::2])
print(a[..., ::2]) # 有...代表全选前面所有维度 ：只能全选一个维度 ...可以全选所有维度

print('-----------------------------')
print(a[0, ...])
print(a[0, ..., :])
print(a[0, :, :])
```

### 高级索引
```python
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
```

### 广播
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407231519075.png)
```python
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
```

### 迭代数组
```python
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
```

### 数组操作
```python
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


# # .flatten() 数组降维，返回折叠后的一维数组，原数组不变
# c = a.flatten()
# print(c)
# # .astype(new_type) 转化数据类型 创建新数组
# print(a)
# x = a.astype(np.float64)
# print(x)
# # .tolist() ndarray数组向列表转换
# x = a.tolist()
# print(x)
```


### ndarray数组的运算
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407221446226.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407221447426.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407221455310.png)

### 数据的CSV文件存取
```python
'''
CSV文件：逗号分隔值文件
np.savetxt(frame, array, fmt='%.18e', delimiter=None)
frame:文件、字符串或产生器 可以是.gz .bz2的压缩文件
array:存入文件的数组
fmt:存入文件的格式 %d %.2f %.18e
delimiter:分割字符串。默认为任何空格
'''
# 整数保存
import numpy as np
a = np.arange(100).reshape(5, 20)
np.savetxt('related_data/test1.csv', a, fmt='%d', delimiter=',')
# 浮点数保存
a = np.arange(100).reshape(5, 20)
np.savetxt('related_data/test2.csv', a, fmt='%.1f', delimiter=',')

'''
np.loadtxt(frame, dtype=np.float, delimiter=None, unpack=False)
frame:文件、字符串或产生器 可以是.gz .bz2的压缩文件
dtype:数据类型可选
delimiter:分割字符串。默认为任何空格
unpack:若为True，读入属性将分别写入不同变量
'''
# 默认浮点型
b = np.loadtxt('related_data/test1.csv', delimiter=',')
print(b)
# 指定整数型
# * 新版numpy无int类型 使用int_
b = np.loadtxt('related_data/test1.csv', dtype=np.int_, delimiter=',')
print(b)

# csv只能有效存储一维和二维数组即load/save函数只能存取一维和二维数组
```

### 多维数据的存取
```python
'''
a.tofile(frame, sep='', format='%s')
frame: 文件、字符串
sep: 数据分割字符串 如果为空串 写入文件为二进制
format: 写入数据格式
'''
import numpy as np

a= np.arange(100).reshape(5, 10, 2)
# 只是逐项输出数据 看不出维度信息
a.tofile("related_data/test1.dat", sep=',', format='%d')
# 存储为二进制文件 占用空间更小 如果可以知道显示字符的编码以及字节之间的关系就可以实现转化
a.tofile("related_data/test2.dat", format='%d')

'''
np.fromfile(frame, dtype=float, count=-1, sep='')
frame:文件、字符串
dtype:读取的数据类型
count:读入元素个数,-1表示读入全部文件
sep: 数据分割字符串 如果为空串 写入文件为二进制

读取时需要知道存入文件时数组的维度和元素类型
可以通过再写一个元数组文件存储维度和元素类型信息
'''
# 文本文件
b = np.fromfile('related_data/test1.dat', dtype=np.int_, sep=',')
print(b)
b = np.fromfile('related_data/test1.dat', dtype=np.int_, sep=',').reshape(5, 10, 2)
print(b)

# 二进制文件 无需指定分隔符
c = np.fromfile("related_data/test2.dat", dtype=np.int_).reshape(5, 10 ,2)
print(c)

'''
NumPy便捷文件存取 固定文件格式
正常文件：np.save(fname, array)   压缩：np.savez(fname, array)
fname: 文件名 以.npy为扩展名 压缩扩展名为.npz
array: 数组变量

np.load(fname)
fname: 文件名 以.npy为扩展名 压缩扩展名为.npz

可以直接还原数组维度和元素类型信息
因为在文件开头用显示的方式 将数组的源信息存储在了第一行
'''
np.save("related_data/test1.npy", a)
d = np.load("related_data/test1.npy")
```

### NumPy的随机数函数
```python
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
```

### NumPy的统计函数
```python
import numpy as np

a = np.arange(15).reshape(3, 5)
print(a)

# sum(a, axis=None) 计算数组a的axis轴所有元素之和 默认全部轴
print(np.sum(a))

# mean(a, axis=None) 计算数组a的axis轴所有元素期望 默认全部轴
print(np.mean(a, axis=0))
print(np.mean(a, axis=1))

# average(a, axis=None, weights=None) 计算数组a的axis轴所有元素加权平均值 默认全部轴
print(np.average(a, axis=0, weights=[10, 5, 1]))

# std(a, axis=None) 计算数组a的axis轴所有元素标准差 默认全部轴
print(np.std(a))

# var(a, axis=None) 计算数组a的axis轴所有元素方差 默认全部轴
print(np.var(a))

a = np.arange(15, 0, -1).reshape(3, 5)

# min(a) max(a) 最小值最大值
print(np.min(a), np.max(a))

# argmin(a) argmax(a) a中元素最小值、最大值的降到一维后下标
print(np.argmin(a), np.argmax(a))

# unravel_index(index, shape) 根据shape将一维下标index转多维下标
print(np.unravel_index(np.argmin(a), a.shape))

# ptp(a) a中元素最大值和最小值之差
print(np.ptp(a))

# median(a) 元素中位数
print(np.median(a))
```

### NumPy的梯度函数
```python
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
```

### 图像的数组表示
图像是一个由像素组成的二维矩阵，每个元素是一个RGB值
```python
from PIL import Image
import numpy as np

im = np.array(Image.open("related_data/dog.jpg"))
# 图像是一个三维数组，维度分别是高度、宽度和像素RGB值
print(im.shape, im.dtype)
```

### 图像的变换
```python
from PIL import Image
import numpy as np

# 像素值翻转
a = np.array(Image.open("related_data/dog.jpg"))
print(a.shape, a.dtype)
b = [255, 255, 255] - a
# Image.fromarray 数组转图像Image对象
im = Image.fromarray(b.astype('uint8'))
im.save("related_data/dog_reverse_trans.jpg")

# .convert('L') 将彩色图片转换灰度值图片 灰度值翻转
a = np.array(Image.open("related_data/dog.jpg").convert('L'))
b = 255 - a
im = Image.fromarray(b.astype('uint8'))
im.save('related_data/dog_gray_trans.jpg')

# 区间变换
c = (100/255)*a + 150
im = Image.fromarray(c.astype('uint8'))
im.save('related_data/dog_interval_trans.jpg')

# 像素平方
d = 255 * (a/255) ** 2
im = Image.fromarray(d.astype('uint8'))
im.save('related_data/dog_square_trans.jpg')
```

### 实例_图像的手绘效果
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407230152126.png)
```python
# 手绘效果：黑白灰色、边界线条较重、相同或相近色彩趋于白色、略有光源效果
from PIL import Image
import numpy as np

a = np.array(Image.open('related_data/dog.jpg').convert('L')).astype('float')

# 调整图像明暗和添加虚拟深度
depth = 10. # 预设深度值为10 取值范围0-100
grad = np.gradient(a)
grad_x, grad_y = grad
grad_x = grad_x * depth / 100. # 根据深度调整x和y方向的梯度值 除以100进行归一化
grad_y = grad_y * depth / 100.
A = np.sqrt(grad_x**2 + grad_y**2 + 1.) # 构建x和y轴梯度的三维归一化单位坐标系
uni_x = grad_x / A # 单位法向量
uni_y = grad_y / A
uni_z = 1. / A

vec_el = np.pi / 2.2  
vec_az = np.pi / 4.
dx = np.cos(vec_el) * np.cos(vec_az) # .cos(vec_el)单位光线在平面上的投影长度  dxdydz光源对xyz三方向的影响程度
dy = np.cos(vec_el) * np.sin(vec_az)
dz = np.sin(vec_el)

# 梯度和光源相互作用，将梯度转化为灰度
b = 255 * (dx * uni_x + dy * uni_y + dz * uni_z)
# 避免数据越界 将生成的灰度值裁剪至0-255区间
b: np.ndarray = b.clip(0, 255)

im = Image.fromarray(b.astype('uint8'))
im.save('related_data/dog_hand_painting.jpg')
```

## Matplotlib
### Matplotlb库的介绍
### pyplot的plot()函数
### pyplot的中文显示
### pyplot的文本显示
### pyplot的子绘图区域
### pyplot基础图表函数概述
### pyplot饼图的绘制
### pyplot直方图的绘制
### pyplot极坐标的绘制
### pyplot散点图的绘制
### 实例分析_引力波的绘制
## Pandas
### Pandas库的Series类型
### Pandas库的DataFrame类型
### Pandas库的数据类型操作
### Pandas库的数据类型运算
### 数据的排序
### 数据的基本统计分析
### 数据的累计统计分析
### 数据的相关分析
