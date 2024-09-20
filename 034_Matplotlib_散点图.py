import matplotlib.pyplot as plt
import numpy as np

'''
scatter(x, y, s=None, c=None, marker=None, camap=None, norm=None, vmin=None, vmax=None, alpha=None,
        linewidths=None, *, edgecolors=None, plotnonfinite=False, data=None, **kwargs)
x, y:长度相同的数组。数据点
s:点的大小，默认20，也可以是数组，数组每个参数为对应点的大小
c:点的颜色，默认蓝色，也可以是RGB或RGBA二维行数组
marker:点的样式，默认o
cmap:colormap 默认None，标量或是一个colormap的名字，只有c是一个浮点数数组时才使用，如果没有申明就是image.cmap
norm:归一化 默认None，数据亮度在0-1之间，只有c是一个浮点数数组才使用
vmin,vmax:亮度设置，在norm参数存在时会忽略
alpha:透明度设置，0-1之间，默认None即不透明
linewidth:标记点长度
edgecolors:颜色或颜色序列，可选'face'(默认) 'none' None
plotnonfinite:布尔值，设置是否使用非限定的c(inf,-inf或nan)绘制点
**kwargs:其他参数
'''

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([1, 4, 9 ,16, 7, 11, 23, 18])

plt.scatter(x, y)
plt.show()

# 设置图标大小
sizes = np.array([20, 50, 100, 200, 500, 1000, 60, 90])
plt.scatter(x, y, s=sizes)
plt.show()

# 自定义点的颜色
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([1, 4, 9 ,16, 7, 11, 23, 18])
plt.scatter(x, y, color='hotpink')
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([20, 12, 62, 12, 52, 67, 10, 5])
plt.scatter(x, y, color='#88C999')
plt.show()

# 随机数设置散点图
np.random.seed(19680801) # 随机数生成器种子
N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = (30 * np.random.rand(N)) ** 2
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.title('test')
plt.show()

# 颜色条colormap 像一个颜色列表，每种颜色有一个范围从0到100的值
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
y = np.array([1, 4, 9 ,16, 7, 11, 23, 18, 20, 2, 18])
colors = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.scatter(x, y, c=colors, cmap='viridis') # cmap参数默认viridis
plt.show()
# 要显示颜色条，使用plt.colorbar()方法
plt.scatter(x, y, c=colors, cmap='viridis')
plt.colorbar()
plt.show()
# 换个颜色条参数 设置afmhot_r 带r即为颜色条翻转 参数选择见学习笔记
plt.scatter(x, y, c=colors, cmap='afmhot')
plt.colorbar()
plt.show()
plt.scatter(x, y, c=colors, cmap='afmhot_r')
plt.colorbar()
plt.show()