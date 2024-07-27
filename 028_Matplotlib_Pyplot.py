'''
Pyplot是Matplotlib的子库，提供了和matlab类似的绘图API
是常用的绘图模块，包含一系列绘图函数的相关函数，可以对图形进行修改
'''

import matplotlib.pyplot as plt
import numpy as np

# 利用两点绘制一条线
xpoints = np.array([0, 6])
ypoints = np.array([0, 100])
plt.plot(xpoints, ypoints)
plt.show()

'''
plot() 绘制线图和散点图
画单条线 plot([x], y. [fmt], *, data=None, **kwargs)
画多条线 plot([x], y, [fmt], [x2], y2, [fmt2], ..., **kwargs)
x,y: 点或线的节点 x为x轴数据，y为y轴数据，数据可以是列表或数组
fmt: 可选，定义基本格式（颜色、标记和线条样式等）
**kwargs: 可选。用在二维平面图上，设置指定属性，如标签，线的宽度等

标记字符：. 点标记  , 像素标记(极小点)  o 实心圈标记  v 倒三角标记  ^ 上三角标记  > 右三角标记  < 左三角标记等
颜色字符：b m:洋红色 g y r k:黑色 w c '#0080000'RGB颜色符串  多条曲线不指定颜色，会自动选择不同颜色
线型参数：- 实线  -- 破折线  -. 点划线  : 虚线
'''

# 绘制(1, 3)到(8, 10)的线
xpoints = np.array([1, 8])
ypoints = np.array([3, 10])
plt.plot(xpoints, ypoints)
plt.show()

# 绘制两个坐标点，而不是一条线，传参o
xpoints = np.array([1, 8])
ypoints = np.array([3, 10])
plt.plot(xpoints, ypoints, 'o')
plt.show()

# 绘制一条不规则线
xpoints = np.array([1, 2, 6, 8])
ypoints = np.array([3, 8, 1, 10])
plt.plot(xpoints, ypoints)
plt.show()

# 如果不指定x轴的点。x会根据y的值来设置为0,1,2,3,..N-1
ypoints = np.array([3, 10])
plt.plot(ypoints)
plt.show()

# 若y值更多一点
ypoints = np.array([3, 8, 1, 10, 5, 7])
plt.plot(ypoints)
plt.show()

# 正弦余弦图 x,y对应正弦 x,z对应余弦
x = np.arange(0, 4 * np.pi, 0.1)
y = np.sin(x)
z = np.cos(x)
plt.plot(x, y, x, z)
plt.show()