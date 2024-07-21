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
