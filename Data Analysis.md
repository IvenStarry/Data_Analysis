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
ndarray对象属性：
1. .ndim 轴的数量或维度的数量
2. .shape 对象的尺度，对于矩阵 n行m列
3. .size 对象元素的个数 相当于n*m
4. .dtype 元素类型
5. .itemsize 对象的每个元素的大小，以字节为单位
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407211328315.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407211329907.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407211329850.png)
python仅支持整数、浮点数和复数3种类型
科学计算设计数据较多，对存储和性能有较高要求，对元素类型精确定义有助于NumPy合理使用存储空间并优化性能，也有助于程序员对程序规模合理评估
```python
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
```

### ndarray数组的创建和变换
```python
import numpy as np

# todo 1.从python的列表、元组等类型创建adarray数组
# x = np.array(list/tuple, dtype='int32') 可以指定数据类型
# 从列表
x = np.array([0, 1, 2, 3], dtype=np.float64)
print(x)
# 从元组
x = np.array((0, 1, 2, 3))
print(x)
# 从列表和元组混合类型创建
x = np.array(([0, 1], [1, 2], [2, 3], [3, 4]))
print(x)

# todo 2.使用NumPy中函数创建adarray数组
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
# np.ones_like(a) 根据数组a的形状生成全1数组
x = np.ones_like(a)
print(x)
# np.zeros_like(a) 根据数组a的形状生成全0数组
x = np.zeros_like(a)
print(x)
# np.full_like(a, val) 根据数组a的形状生成全val数组
x = np.full_like(a, 5)
print(x)

# todo 3.使用NumPy中其他函数创建adarray数组
# np.linspace(start, end, num) num数组元素个数 4个元素有三个间隔 间隔为(10-1)/3=3
a = np.linspace(1, 10, 4)
print(a)
# 若设置endpoint则不以end这个数结尾 但仍生成4个数 5个元素（包括10）有四个间隔 间隔变为(10-1)/4=2.25
b = np.linspace(1, 10, 4, endpoint=False)
print(b)
# np,concatenate() 两个或多个数组合并成一个新的数组
x = np.concatenate((a, b))
print(x)

# todo ndarray数组的变换
a = np.ones((2, 3, 4), dtype=np.int32)
print(a)
# .reshape(shape) 返回新数组
x = a.reshape(3, 8)
print(x)
# .resize(shape) 修改原数组
a.resize(4, 6)
print(a)
# .swapaxes(ax1, ax2) 将数组的n个维度的2个维度调换 返回新数组 类似转置
a = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
print(a)
b = a.swapaxes(0, 1)
print(b)
# .flatten() 数组降维，返回折叠后的一维数组，原数组不变
c = a.flatten()
print(c)
# .astype(new_type) 转化数据类型 创建新数组
print(a)
x = a.astype(np.float64)
print(x)
# .tolist() ndarray数组向列表转换
x = a.tolist()
print(x)
```

### ndarray数组的操作
```python
'''
索引：获取数组中特定位置的元素
切片：获取数组元素子集的过程
'''
import numpy as np
# 一维数组
a = np.array([11, 22, 33, 44, 55])
print(a[2])
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
```

### ndarray数组的运算
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407221446226.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407221447426.png)
![](https://cdn.jsdelivr.net/gh/IvenStarry/Image/MarkdownImage/202407221455310.png)
```python
import numpy as np
a = np.arange(24).reshape((2, 3, 4))
print(a)

# 数组与标量之间的运算作用于数组的每一个元素
a.mean()
a = a / a.mean()
print(a)

a = np.arange(24).reshape((2, 3, 4))
print(f"平方运算:{np.square(a)}")
print(f"开方运算：{np.sqrt(a)}")
print(f"整数小数分离{np.modf(np.sqrt(a))}") # np.modf()将整数和小数部分分成两个部分

b = np.sqrt(a)
print(a)
print(b)
print(np.maximum(a, b)) # 输出结果浮点数
print(a > b)
```

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
