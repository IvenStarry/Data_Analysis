import matplotlib.pyplot as plt
import numpy as np

# grid() 设置图表中的网格线
'''
plt.grid(b=None, which='major', axis='both', )
b: 可选，默认None，可设置bool值，true显示网格线，false不显示，如果设置**kwargs参数，值为true
which: 可选，可选值有'major'(默认),'minor','both' 表示应用更改的网格线
axis: 可选，设置显示哪个方向的网格线，可选'both'(默认) 'x','y'，分别表示x轴y轴两个方向
**kwargs: 可选，设置网格样式，可以是color='r',linestyle='-'和linewidth=2，分别表示网格线的颜色，样式和宽度
'''
x = np.array([1, 2, 3, 4])
y = np.array([1, 4, 9, 16])
# 网格线参数默认值
plt.title('grid test')
plt.xlabel('x-label')
plt.ylabel('y-label')
plt.plot(x, y)
plt.grid()
plt.show()

# 网格线参数设置axis
plt.title('grid test')
plt.xlabel('x-label')
plt.ylabel('y-label')
plt.plot(x, y)
plt.grid(axis='x') # 设置x轴方向显示网格线
plt.show()

# 设置网格线的样式 样式同绘图线类型标记，颜色标记，线宽度
# grid(color='color', linestyle='linestyle', linewidth='linewidth')
plt.title('grid test')
plt.xlabel('x-label')
plt.ylabel('y-label')
plt.plot(x, y)
plt.grid(color='r', linestyle='--', linewidth=0.5) # 设置x轴方向显示网格线
plt.show()