import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile # 读取波形文件的库

rate_h, hstrain = wavfile.read(r'related_data/H1_Strain.wav', 'rb') # 读取声音文件
rate_l, lstrain = wavfile.read(r'related_data/L1_Strain.wav', 'rb')
# 读取时间序列和信号数据 genfromtxt执行两个循环 第一个循环将文件每一行转化为字符串 第二个循环将每个字符串转换成相应的类型 因为读取出的是一个两行的矩阵 不方便使用 因此使用tranpose转置
reftime, ref_H1 = np.genfromtxt('related_data/wf_template.txt').transpose()

htime_interval = 1 / rate_h
ltime_interval = 1 / rate_l

htime_len = hstrain.shape[0] / rate_h
htime = np.arange(-htime_len / 2, htime_len / 2, htime_interval)
ltime_len = lstrain.shape[0] / rate_l
ltime = np.arange(-ltime_len / 2, ltime_len / 2, ltime_interval)

fig = plt.figure(figsize=(12, 6))

plth = fig.add_subplot(221)
plth.plot(htime, hstrain, 'y')
plth.set_xlabel('Time(seconds)')
plth.set_ylabel('H1 Strain')
plth.set_title('H1 Strain')

pltl = fig.add_subplot(222)
pltl.plot(ltime, lstrain, 'g')
pltl.set_xlabel('Time(seconds)')
pltl.set_ylabel('L1 Strain')
pltl.set_title('L1 Strain')

plth = fig.add_subplot(212)
plth.plot(reftime, ref_H1)
plth.set_xlabel('Time(seconds)')
plth.set_ylabel('Template Strain')
plth.set_title('Template Strain')
fig.tight_layout() # 自动调整外部边缘

plt.savefig('related_data/Gravitational_Waves_Original.png')
plt.show()
plt.close(fig)