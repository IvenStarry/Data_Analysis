import matplotlib.font_manager
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 获取系统字体库列表
# from matplotlib import pyplot as plt
# import matplotlib
# a=sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])

# for i in a:
#     print(i)


# 两种方法
# 1.替换Matplotlib默认字体
plt.rcParams['font.family'] = 'Source Han Sans CN'
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
plt.plot(x, y)
plt.title('折线图实例(替换默认字体)')
plt.xlabel('x轴')
plt.ylabel('y轴')
plt.show()

# 2. 使用OTF字体库
font = matplotlib.font_manager.FontProperties(fname='related_data/SourceHanSansCN-Bold.otf')
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
plt.plot(x, y)
plt.title('折线图实例(设置字体属性)', fontproperties=font)
plt.xlabel('x轴', fontproperties=font)
plt.ylabel('y轴', fontproperties=font)
plt.show()
