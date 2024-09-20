# 手绘效果：黑白灰色、边界线条较重、相同或相近色彩趋于白色、略有光源效果
from PIL import Image
import numpy as np

a = np.array(Image.open('related_data/dog.jpg').convert('L')).astype('float')

# 调整图像明暗和添加虚拟深度
depth = 10. # 预设深度值为10 取值范围0-100
grad = np.gradient(a)
grad_x, grad_y = grad
grad_x = grad_x * depth / 100. # 根据深度调整x和y方向的梯度值 除以100进行归一化
grad_y = grad_y * depth / 100.
A = np.sqrt(grad_x**2 + grad_y**2 + 1.) # 构建x和y轴梯度的三维归一化单位坐标系
uni_x = grad_x / A # 单位法向量
uni_y = grad_y / A
uni_z = 1. / A

vec_el = np.pi / 2.2  
vec_az = np.pi / 4.
dx = np.cos(vec_el) * np.cos(vec_az) # .cos(vec_el)单位光线在平面上的投影长度  dxdydz光源对xyz三方向的影响程度
dy = np.cos(vec_el) * np.sin(vec_az)
dz = np.sin(vec_el)

# 梯度和光源相互作用，将梯度转化为灰度
b = 255 * (dx * uni_x + dy * uni_y + dz * uni_z)
# 避免数据越界 将生成的灰度值裁剪至0-255区间
b: np.ndarray = b.clip(0, 255)

im = Image.fromarray(b.astype('uint8'))
im.save('related_data/dog_hand_painting.jpg')