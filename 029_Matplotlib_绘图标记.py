import matplotlib.pyplot as plt
import matplotlib.markers
import numpy as np

# 自定义标记 plot()方法的marker参数
ypoints = np.array([1, 3, 4, 5, 8, 9, 6, 1, 3, 4, 5, 2, 4])
plt.plot(ypoints, marker='o')
plt.show()

# marker定义的符号 具体见学习笔记
plt.plot(ypoints, marker='*')
plt.show()

# 定义下箭头
plt.plot([1, 2, 3], marker=matplotlib.markers.CARETDOWN)
plt.show()
plt.plot([1, 2, 3], marker=7) # 看表也可以用7
plt.show()

# fmt参数 定义了基本格式，如标记、线条样式和颜色
# fmt = '[marker][line][color]'
ypoints = np.array([6, 2, 13, 10])
plt.plot(ypoints, 'o:r') # o实心圆标记 :虚线 r红色
plt.show()

# 标记大小与颜色
'''
markersize:        ms     标记的大小
markerfacecolor:   mfc    定义标记内部的颜色
markeredgecolor:   mec    定义标记边框的颜色
'''
ypoints = np.array([6, 2, 13, 10])
plt.plot(ypoints, marker='o', ms=20, mfc='w', mec='r')
plt.show()
plt.plot(ypoints, marker='o', ms=10, mfc='#4CAF50', mec='#4CAF50') # 自定义颜色
plt.show()