# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread(r'C:\Users\kilok\Desktop\absfbsxt.jpg')            # 打开jpg格式文件
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                        # 转化为灰度图
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)    # OTSU二值化处理

# noise removal                                                     # 噪声移除
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))           # 获取一个3x3的矩形结构
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)            # 传入上面获取的thresh图片，并进行开运算（先腐蚀再膨胀）kernel为腐蚀操作的核，iterations为腐蚀操作的次数

sure_bg = cv2.dilate(opening, kernel, iterations=2)  # sure background area         # 对opening图像进行膨胀操作，使用kernel迭代两次
sure_fg = cv2.erode(opening, kernel, iterations=2)  # sure foreground area          # 对opening图像进行腐蚀操作，使用kernel迭代两次
unknown = cv2.subtract(sure_bg, sure_fg)  # unknown area                            # 减法，用膨胀图减去腐蚀图，得到边缘轮廓

cv2.imshow('unknow',unknown)

# Perform the distance transform algorithm
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)                     # 距离提取，即计算各个像素点距离背景0的距离，也即海平面高度
# Normalize the distance image for range = {0.0, 1.0}
cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)              # 对距离进行归一化，0到1之间

# Finding sure foreground area
ret, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)      # 确定合适的前景图

# Finding unknown region
sure_fg = np.uint8(sure_fg)                                                         # 将sure_fg转为uint8格式
unknown = cv2.subtract(sure_bg,sure_fg)                                             # 这一部分在余下的不确定部分里寻找边界

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)                                     # 用0标记图像的背景，用大于0的整数标记其他对象
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1                                                                 # 见上行注释
# Now, mark the region of unknown with zero
markers[unknown==255] = 0                                                           # 将边缘部分标为0

# 使用分水岭算法执行基于标记的图像分割，将图像中的对象与背景分离
markers = cv2.watershed(img, markers)
img[markers==-1] = [0,0,255]                                                        # 将边界标记为红色

plt.imshow(img)
input()