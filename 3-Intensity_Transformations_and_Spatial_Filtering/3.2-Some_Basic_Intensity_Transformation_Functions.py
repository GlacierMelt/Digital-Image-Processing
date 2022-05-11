import cv2
import numpy as np


# IMAGE NEGATIVES（图像反转）
img = cv2.imread('./images/DIP3E_Original_Images_CH03/Fig0304(a)(breast_digital_Xray).tif')
print('Shape Of Image:', img.shape)

L = 256 - 1
img_neg = L - img
img_neg = np.hstack([img, img_neg])
cv2.imshow('img_neg', img_neg)
cv2.waitKey(0)
cv2.destroyAllWindows()


# LOG TRANSFORM（对数变换）
img = cv2.imread('images/DIP3E_Original_Images_CH03/Fig0305(a)(DFT_no_log).tif')

c = 255 / np.log(1 + np.max(img))
img_log = c * np.log(1 + img)
img_log = np.array(img_log, dtype = np.uint8)   # 指定数据类型, 以便float值将转换为int
imgs= np.hstack([img, img_log])

cv2.imshow('img_log', imgs)
cv2.waitKey(0)
cv2.destroyAllWindows()