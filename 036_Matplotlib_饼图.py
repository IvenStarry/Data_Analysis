import matplotlib.pyplot as plt
import numpy as np

'''
pie(x, explode=None, labels=None, colors=None, autopct=None, pctdistance=0.6, shadow=False, labelsiatance=1.1, startangle=0,
    radius=1, counterclock=True, wedgeprops=None, center=0, 0, frame=False, totatelabels=False, *, normalize=None, data=None)[source]

x：浮点型数组或列表，用于绘制饼图的数据，表示每个扇形的面积。
explode：数组，表示各个扇形之间的间隔，默认值为0。
labels：列表，各个扇形的标签，默认值为 None。
colors：数组，表示各个扇形的颜色，默认值为 None。
autopct：设置饼图内各个扇形百分比显示格式，%d%% 整数百分比，%0.1f 一位小数， %0.1f%% 一位小数百分比， %0.2f%% 两位小数百分比。
labeldistance：标签标记的绘制位置，相对于半径的比例，默认值为 1.1，如 <1则绘制在饼图内侧。
pctdistance：：类似于 labeldistance，指定 autopct 的位置刻度，默认值为 0.6。
shadow：：布尔值 True 或 False，设置饼图的阴影，默认为 False，不设置阴影。
radius：：设置饼图的半径，默认为 1。
startangle：：用于指定饼图的起始角度，默认为从 x 轴正方向逆时针画起，如设定 =90 则从 y 轴正方向画起。
counterclock：布尔值，用于指定是否逆时针绘制扇形，默认为 True，即逆时针绘制，False 为顺时针。
wedgeprops ：property 字典类型，默认值 None。用于指定扇形的属性，比如边框线颜色、边框线宽度等。例如：wedgeprops={'linewidth':5} 设置 wedge 线宽为5。
textprops ：property 字典类型，用于指定文本标签的属性，比如字体大小、字体颜色等，默认值为 None。
center ：浮点类型的列表，用于指定饼图的中心位置，默认值：(0,0)。
frame ：布尔类型，用于指定是否绘制饼图的边框，默认值：False。如果是 True，绘制带有表的轴框架。
rotatelabels ：布尔类型，用于指定是否旋转文本标签，默认为 False。如果为 True，旋转每个 label 到指定的角度。
data：用于指定数据。如果设置了 data 参数，则可以直接使用数据框中的列作为 x、labels 等参数的值，无需再次传递。

除此之外，pie() 函数还可以返回三个参数：
1. wedges：一个包含扇形对象的列表。
2. texts：一个包含文本标签对象的列表。
3. autotexts：一个包含自动生成的文本标签对象的列表。
'''

y = np.array([35, 25, 25, 15])
plt.pie(y)
plt.show()

# 设置饼图各个扇区的标签和颜色
plt.pie(y, labels=['A', 'B', 'C', 'D'], colors=['#D5695D', '#5D8CA8', '#65A479', '#A564C9'])
plt.title('pie test')
plt.show()

# 突出显示第二个扇形，并格式化输出百分比
sizes = [15, 30, 45, 10]
labels = ['A', 'B', 'C', 'D']
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
explode = (0, 0.1, 0, 0) # 突出显示
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.show()

sizes = [35, 25, 25, 15]
labels = ['A', 'B', 'C', 'D']
colors = ['#D5695D', '#5D8CA8', '#65A479', '#A564C9']
explode = (0, 0.2, 0, 0) # 突出显示
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%.2f%%')
plt.show()