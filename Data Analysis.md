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
### ndarray数组的运算
### 数据的CSV文件存取
### 多维数据的存取
### NumPy的随机数函数
### NumPy的统计函数
### NumPy的梯度函数
### 图像的数组表示
### 图像的变换
### 实例分析_图像的手绘效果
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
