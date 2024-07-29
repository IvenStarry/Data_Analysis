import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

'''
imshow() 用于显示图像，常用于绘制二维的灰度图像或彩色图像，还可用于绘制矩阵、热力图、地图等
imshow(x, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vimn-None, vmax=None, origin=None, extent=None,
        shape=None, filternorm=1, filterrad=4.0, imlim=None, resample=None, url=None, *, data=None, **kwargs)

X：输入数据。可以是二维数组、三维数组、PIL图像对象、matplotlib路径对象等。
cmap：颜色映射。用于控制图像中不同数值所对应的颜色。可以选择内置的颜色映射，如gray、hot、jet等，也可以自定义颜色映射。
norm：用于控制数值的归一化方式。可以选择Normalize、LogNorm等归一化方法。
aspect：控制图像纵横比（aspect ratio）。可以设置为auto或一个数字。
interpolation：插值方法。用于控制图像的平滑程度和细节程度。可以选择nearest、bilinear、bicubic等插值方法。
alpha：图像透明度。取值范围为0~1。
origin：坐标轴原点的位置。可以设置为upper或lower。
extent：控制显示的数据范围。可以设置为[xmin, xmax, ymin, ymax]。
vmin、vmax：控制颜色映射的值域范围。
filternorm 和 filterrad：用于图像滤波的对象。可以设置为None、antigrain、freetype等。
imlim： 用于指定图像显示范围。
resample：用于指定图像重采样方式。
url：用于指定图像链接。
'''

# 显示灰度图像
img = np.random.rand(10, 10)
plt.imshow(img, cmap='gray')
plt.show()

# 显示彩色图像
img = np.random.rand(10, 10, 3) # 生成size=(10,10,3)的均匀分布数组[0,1)
plt.imshow(img)
plt.show()

# 显示热力图
data = np.random.rand(10, 10)
plt.imshow(data, cmap='hot')
plt.colorbar() # 加颜色条
plt.show()

# 显示地图
img = Image.open('related_data/map.jpeg')
data = np.array(img)
plt.imshow(data)
plt.axis('off') # 隐藏坐标轴
plt.show()

# 显示矩阵
data = np.random.rand(10, 10)
plt.imshow(data)
plt.show()

# 三种不同imshow图像展示
n = 4
a = np.reshape(np.linspace(0, 1, n ** 2), (n, n)) # 创建n*n的二维数组
plt.figure(figsize=(12, 4.5))

# 展示灰度的色彩映射方式 进行没有进行颜色的混合
plt.subplot(131)
plt.imshow(a, cmap='gray', interpolation='nearest')
plt.xticks(range(n))
plt.yticks(range(n))
plt.title('gray color map, no blending', y=1.02, fontsize=12)

# 展示使用viridis颜色映射的图像 没有颜色的混合
plt.subplot(132)
plt.imshow(a, cmap='viridis', interpolation='nearest')
plt.yticks([]) # x轴刻度位置的列表，若传入空列表，即不显示x轴
plt.xticks(range(n))
plt.title('viridis color map, no blending', y=1.02, fontsize=12)

# 展示使用viridis颜色映射的图像 使用双立方插值的方法进行颜色混合
plt.subplot(133)
plt.imshow(a, cmap='viridis', interpolation='bicubic')
plt.yticks([])
plt.xticks(range(n))
plt.title('visidis color map, bicubic blending', y=1.02, fontsize=12)

plt.show()