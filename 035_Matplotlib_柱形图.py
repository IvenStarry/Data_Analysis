import matplotlib.pyplot as plt
import numpy as np

'''
bar(x, height, width=0.8, bottom=None, *, align='center', data=None, **kwargs)
x: 浮点型数组，图形x轴数组
height: 浮点型数组，柱形图高度
width: 浮点型数组，柱形图宽度
bottom: 浮点型数组，底座的y坐标，默认0
align: 柱形图与x坐标的对齐方式,'center'以x位置为中心，这是默认值。'edge'将柱形图左边缘与x位置对齐。要对齐右边缘的条形，可以传递负数的宽度值及align='edge'
**kwargs: 其他参数
'''

x = np.array(['test1', 'test2', 'test3', 'test4'])
y = np.array([10, 20, 30, 40])
plt.bar(x, y)
plt.show()

# 垂直方向的柱形图可以用barh()方法进行设置
plt.barh(x, y)
plt.show()

# 设置柱形图颜色
plt.bar(x, y, color='#4CAF50')
plt.show()

# 自定义各个柱形的颜色
plt.bar(x, y, color=['#4CAF50', 'red', 'hotpink', '#556B2F'])
plt.show()

# 设置柱形图宽度 bar用width barh用height
plt.subplot(121)
plt.bar(x, y, width=0.1)
plt.subplot(122)
plt.barh(x, y, height=0.5)
plt.show()

