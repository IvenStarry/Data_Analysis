import matplotlib.pyplot as plt
import matplotlib.font_manager
import numpy as np

# 设置xy轴的标签 xlabel() ylabel()
x = np.array([1, 2, 3, 4])
y = np.array([1, 4, 9, 16])
plt.plot(x, y)
plt.xlabel('x-label')
plt.ylabel('y-label')
plt.show()

# 标签 title()
plt.plot(x, y)
plt.xlabel('x-label')
plt.ylabel('y-label')
plt.title('matplotlib test')
plt.show()

# 图形中文显示 fontproperties 可以使用系统的字体 这里使用思源黑体
zhfont1 = matplotlib.font_manager.FontProperties(fname='related_data/SourceHanSansCN-Bold.otf')
x = np.arange(1, 11)
y = 2 * x + 5
plt.plot(x, y)
plt.xlabel('x轴', fontproperties=zhfont1)
plt.ylabel('y轴', fontproperties=zhfont1)
plt.title('matplotlib 练习', fontproperties=zhfont1)
plt.show()

# 自定义字体样式 fontdict 设置字体颜色大小
zhfont1 = matplotlib.font_manager.FontProperties(fname='related_data/SourceHanSansCN-Bold.otf', size=18)
font1 = {'color':'blue', 'size':20}
font2 = {'color':'darkred', 'size':20}
plt.title('matplotlib 练习', fontproperties=zhfont1, fontdict=font1)
plt.plot(x, y)
plt.xlabel('x轴', fontproperties=zhfont1)
plt.ylabel('y轴', fontproperties=zhfont1)
plt.show()

# 标题与标签的定位
'''
title() 方法提供loc参数设置标题位置 可以设置为'left','right'和'center',默认center
xlabel()方法提供loc参数设置x 轴位置 可以设置为'left','right'和'center',默认center
ylabel()方法提供loc参数设置y 轴位置 可以设置为'bottom','top'和'center',默认center
'''
plt.title('matplotlib 练习', fontproperties=zhfont1, fontdict=font1, loc='right')
plt.plot(x, y)
plt.xlabel('x轴', fontproperties=zhfont1, loc='left')
plt.ylabel('y轴', fontproperties=zhfont1, loc='bottom')
plt.show()

# 任意位置添加文本 text(x, y, str)
a = np.arange(0.0, 5.0, 0.02)
plt.plot(a, np.cos(2 * np.pi * a), 'r--')
plt.xlabel('横轴：时间', fontproperties='SimHei', fontsize=15, color='green')
plt.ylabel('纵轴：振幅', fontproperties='SimHei', fontsize=15)
plt.title(r'正弦波实例 $y=cos(2 \pi x)$', fontproperties='SimHei', fontsize=25) # latex公式
plt.text(2, 1, r'$ \mu = 100 $')
plt.axis([-1, 6, -2, 2])
plt.grid(True)
plt.show()

# 在图形中增加带箭头的注解 annotate(s, xy=arrow_crd, xytext=text_crd, arrowprop=dict) xy:箭头指向位置 xytext:文本位置
a = np.arange(0.0, 5.0, 0.02)
plt.plot(a, np.cos(2 * np.pi * a), 'r--')
plt.xlabel('横轴：时间', fontproperties='SimHei', fontsize=15, color='green')
plt.ylabel('纵轴：振幅', fontproperties='SimHei', fontsize=15)
plt.title(r'正弦波实例 $y=cos(2 \pi x)$', fontproperties='SimHei', fontsize=25) # latex公式
plt.annotate(r'$\mu = 100$', xy=(2, 1), xytext=(3, 1.5), arrowprops=dict(facecolor='black', shrink=0.1, width=2))
plt.axis([-1, 6, -2, 2])
plt.grid(True)
plt.show()