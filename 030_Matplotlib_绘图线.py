import matplotlib.pyplot as plt
import numpy as np

# 线的类型
# linestyle 简写 ls
ypoints = np.array([6, 2, 13, 10])
plt.plot(ypoints, linestyle='dotted')
plt.show()
plt.plot(ypoints, ls='-.') # 简写
plt.show()

# 线的颜色
# color 简写 c 同绘图标记颜色
plt.plot(ypoints, color='r')
plt.show()
plt.plot(ypoints, c='#8FBC8F') # 简写
plt.show()
plt.plot(ypoints, c='SeaGreen') # 十六进制颜色名
plt.show()

# 线的宽度
# linewidth 简写 lw  值可以是浮点数
plt.plot(ypoints, linewidth='12.5')
plt.show()

# 多条线
# plot()可以包含多对xy值 绘制多条线
y1 = np.array([3, 7, 5, 9])
y2 = np.array([6, 2, 13, 10])
plt.plot(y1) # 未传入x 默认 0123
plt.plot(y2)
plt.show()
x1 = np.array([0, 1, 2, 3])
x2 = np.array([1, 2, 3, 4]) # 自定义坐标
plt.plot(x1, y1, x2, y2)
plt.show()
