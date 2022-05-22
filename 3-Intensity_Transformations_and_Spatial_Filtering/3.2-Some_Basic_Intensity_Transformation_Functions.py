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


# GAMMA TRANSFORM（伽马变换）
img = cv2.imread('images/DIP3E_Original_Images_CH03/Fig0307(a)(intensity_ramp).tif')
gamma = 0.5     # 范围 0.01 ～ 25.0
img_gamma = np.array(255 * (img / 255) ** gamma, dtype = 'uint8')
imgs= np.hstack([img, img_gamma])

cv2.imshow('img_gamma', imgs)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Piecewise Linear Transformation（分段线性变换）
def PiecewiseLinear(img, r1, s1, r2, s2):
    L = 256
    lut = np.zeros(L)

    for pix in range(L):
        if pix <= r1:
            lut[pix] = s1 / r1 * pix
        elif pix <= r2:
            lut[pix] = ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
        else:
            lut[pix] = ((L - 1 - s2) / (L - 1 - r2)) * (pix - r2) + s2

    img_pwl = np.array(cv2.LUT(img, lut), dtype = 'uint8')

    return img_pwl

img = cv2.imread('images/DIP3E_Original_Images_CH03/Fig0320(1)(top_left).tif')
L = 255
# contrast stretching（rmin, rmax)
r1, s1= img.min(), 0
r2, s2 = img.max(), L
## Or contrast stretching using alternative method sigmod replacement（r1, r2, s1, s2)

img_pwl = np.array(PiecewiseLinear(img, r1, s1, r2, s2), dtype = 'uint8')
imgs= np.hstack([img, img_pwl])

cv2.imshow('img_pwl', imgs)
cv2.waitKey(0)
cv2.destroyAllWindows()