from PIL import Image
import numpy as np

# 像素值翻转
a = np.array(Image.open("related_data/dog.jpg"))
print(a.shape, a.dtype)
b = [255, 255, 255] - a
# Image.fromarray 数组转图像Image对象
im = Image.fromarray(b.astype('uint8'))
im.save("related_data/dog_reverse_trans.jpg")

# .convert('L') 将彩色图片转换灰度值图片 灰度值翻转
a = np.array(Image.open("related_data/dog.jpg").convert('L'))
b = 255 - a
im = Image.fromarray(b.astype('uint8'))
im.save('related_data/dog_gray_trans.jpg')

# 区间变换
c = (100/255)*a + 150
im = Image.fromarray(c.astype('uint8'))
im.save('related_data/dog_interval_trans.jpg')

# 像素平方
d = 255 * (a/255) ** 2
im = Image.fromarray(d.astype('uint8'))
im.save('related_data/dog_square_trans.jpg')