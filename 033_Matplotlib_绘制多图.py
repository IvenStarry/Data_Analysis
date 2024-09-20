import matplotlib.pyplot as plt
import numpy as np

# subplot subplots 绘制子图
'''
subplot() 绘图需要指定位置
subplot(nrows, ncols, index, **kwargs)
subplot(pos, **kwargs)
subplot(**kwargs)
subplot(ax)
将绘图区域分为nrows行和ncols列，从左到右从上到下对每个子区域编号1...N，编号可以通过index来设置
suptitle 设置多图的标题
''' 
# 一行二列多图
xpoints = np.array([0, 6])
ypoints = np.array([0, 100])
plt.subplot(1, 2, 1)
plt.plot(xpoints, ypoints)
plt.title('plot 1')

x = np.array([1, 2, 3, 4])
y = np.array([1, 4, 9, 16])
plt.subplot(1, 2, 2)
plt.plot(x, y)
plt.title('plot 2')

plt.suptitle('subplot test')
plt.show()

# 二行二列多图
xpoints = np.array([0, 6])
ypoints = np.array([0, 100])
plt.subplot(2, 2, 1)
plt.plot(xpoints, ypoints)
plt.title('plot 1')

x = np.array([1, 2, 3, 4])
y = np.array([1, 4, 9, 16])
plt.subplot(2, 2, 2)
plt.plot(x, y)
plt.title('plot 2')

x = np.array([1, 2, 3, 4])
y = np.array([5, 16, 17, 8])
plt.subplot(2, 2, 3)
plt.plot(x, y)
plt.title('plot 3')

x = np.array([1, 2, 3, 4])
y = np.array([10, 2, 23, 4])
plt.subplot(2, 2, 4)
plt.plot(x, y)
plt.title('plot 4')

plt.suptitle('subplot test')
plt.show()

'''
subplots() 绘图一次生成多个，在调用时只需要调用生成对象的ax即可
subplots(nrows=1, ncols=1, *, sharex=False, sharey=False, squeeze=True, 
        subplt_kw=None, gridspec_kw=None, **fig_kw)
nrows, ncols 设置图表的函数和列数
sharex, sharey 设置xy轴是否共享属性，可设置none, all, row, col  当为false或none时每个子图的x轴和y轴是独立的
squeeze: bool 默认True 表示额外的维度从返回的Axes(轴)对象中挤出，对于N*1或1*N个子图，返回一个1维数组，对于N*M，N>1和M>1返回一个二维数组，
        如果设置False，则不进行挤压操作，返回一个元素为的Axes的2维，即使最终是1*1
subplt_kw: 可选，字典 字典关键字传递给add_subplot()创建子图
gridspec_kw: 可选，字典 字典关键字传递给GridSpec构造函数创建子图放在网格里grid
**fig_kw: 把详细的关键字传给figure函数
suptitle 设置多图的标题
''' 
x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

# 创建一个图像和子图
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('simple plot')

# 创建2个子图
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(x, y)
ax1.set_title('sharing y axis')
ax2.scatter(x, y)

# 创建4个子图
fig, axs = plt.subplots(2, 2, subplot_kw=dict(projection='polar')) # 投影类型：极坐标图 通过subplot_kw传递给add_subplot中的projection参数
axs[0, 0].plot(x, y)
axs[1, 1].scatter(x, y)

# 共享x轴
plt.subplots(2, 2, sharex='col')

# 共享y轴
plt.subplots(2, 2, sharey='row')

# 共享x和y轴 两种方法
plt.subplots(2, 2, sharex='all', sharey='all')
plt.subplots(2, 2, sharex=True, sharey=True)

# 创建标识为10的图，已存在的则删除
fig, ax = plt.subplots(num=10, clear=True)
plt.show()

'''
总结：subplot和subplots的原理
fig = plt.figure()  #首先调用plt.figure()创建了一个**画窗对象fig**
ax = fig.add_subplot(111)  #然后再对fig创建默认的坐标区（一行一列一个坐标区）  笛卡尔坐标系
#这里的（111）相当于（1，1，1），当然，官方有规定，当子区域不超过9个的时候，我们可以这样简写
'''

'''
plt.subplot2grid(shape, location, colspan=1, rowspan=1)
设定网络 选中网格 确定选中行列区域数量 编号从0开始
shape：把该参数值规定的网格区域作为绘图区域；
location：在给定的位置绘制图形，初始位置 (0,0) 表示第1行第1列；
rowsapan/colspan：这两个参数用来设置让子区跨越几行几列。
'''

#使用 colspan指定列，使用rowspan指定行
a1 = plt.subplot2grid((3,3),(0,0),colspan = 2)
a2 = plt.subplot2grid((3,3),(0,2), rowspan = 3)
a3 = plt.subplot2grid((3,3),(1,0),rowspan = 2, colspan = 2)

x = np.arange(1,10)
a2.plot(x, x*x)
a2.set_title('square')
a1.plot(x, np.exp(x))
a1.set_title('exp')
a3.plot(x, np.log(x))
a3.set_title('log')
plt.tight_layout()
plt.show()