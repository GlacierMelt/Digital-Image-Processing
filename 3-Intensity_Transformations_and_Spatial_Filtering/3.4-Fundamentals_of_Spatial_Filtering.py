import cv2
from cv2 import cvtColor
import numpy as np
import matplotlib.pyplot as plt


# img = np.array([
#     [1, 1, 1],
#     [1, 5, 1],
#     [1, 1, 1]
# ])

# K = np.array([
#     [4, 8,  12],
#     [5, 10, 15],
#     [6, 12, 18]
# ])

# def filter(img, kernel):

#     pad = kernel.shape[0] // 2
#     img_cp = np.copy(img)
#     img_cp = cv2.copyMakeBorder(img_cp, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
#     img_op = np.copy(img)

#     for x in range(img_cp.shape[0] - 2 * pad):
#         for y in range(img_cp.shape[1] - 2 * pad):
#             C = 0
#             img_sl = img_cp[x: x + 2 * pad + 1, y: y + 2 * pad + 1]
#             for m in range(img_sl.shape[0]):
#                 for n in range(kernel.shape[1]):
#                     C += kernel[m][n] * img_sl[m][n] 
#             img_op[x, y] = C

#     return img_op

# print('Origin:\n', img)
# print('\nNo Speed: ')
# print(filter(img, K))

# def Separable_Filter_Kernels(img, kernelX, kernelY):

#     pad = kernelX.shape[0] // 2
#     img_cp = np.copy(img)
#     img_cp = cv2.copyMakeBorder(img_cp, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
#     img_op = np.copy(img_cp)

#     for x in range(img_cp.shape[0]):
#         for y in range(img_cp.shape[1] -  2 * pad):
#             img_sl = img_cp[x, y: y + 2 * pad + 1]
#             C = 0
#             for n in range(kernelX.shape[0]):
#                 C += kernelX[n] * img_sl[n]
#             img_op[x, y + pad] = C

#     for x in range(img_op.shape[0] - 2 * pad):
#         for y in range(pad, img_op.shape[1] -  pad):
#             img_sl = img_op[x: x+ 2 * pad + 1, y]
#             C = 0
#             for m in range(kernelY.shape[0]):
#                 C += kernelY[m] * img_sl[m]
#             img_cp[x+ pad, y] = C

#     return img_cp[pad:-pad, pad:-pad]

# print('\nSeparable Filter: ')
# print(Separable_Filter_Kernels(img,  np.array([1, 2, 3]), np.array([4, 5, 6])))



# # Lowpass Gaussian Filter Kernels
# def GaussianKernel(size, K, sigma, twoDimensional=True):
#     if twoDimensional:
#         kernel = np.fromfunction(lambda x, y: K * np.exp((-1*((x - (size - 1) // 2)**2 
#                                 + (y - (size - 1) // 2)**2)) 
#                                 / (2 * sigma**2)), (size, size))
#     else:
#         kernel = np.fromfunction(lambda x: K * np.exp((-1 * (x - (size - 1) // 2)**2 
#                                 / (2 * sigma**2))), (size,))
#     return kernel / np.sum(kernel)


# def gaussian_filter(img, ks, K, sigma):

#     pad = ks // 2
#     img_cp = np.copy(img)
#     img_cp = cv2.copyMakeBorder(img_cp, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
#     img_op = np.copy(img_cp)

#     # 生成一维高斯滤波器核
#     kernel = GaussianKernel(ks, K, sigma, False)

#     for x in range(img_cp.shape[0]):
#         for y in range(img_cp.shape[1] -  2 * pad):
#             img_sl = img_cp[x, y: y + 2 * pad + 1]
#             C = 0
#             for n in range(ks):
#                 C += kernel[n] * img_sl[n]
#             img_op[x, y + pad] = C

#     for x in range(img_op.shape[0] - 2 * pad):
#         for y in range(pad, img_op.shape[1] -  pad):
#             img_sl = img_op[x: x+ 2 * pad + 1, y]
#             C = 0
#             for m in range(ks):
#                 C += kernel[m] * img_sl[m]
#             img_cp[x+ pad, y] = C

#     return img_cp[pad:-pad, pad:-pad]

# print('\n1D Guassian Filter Kernel: ', GaussianKernel(3, 1, 1, False))
# print('Guassian Filter: ')
# print(gaussian_filter(img, 3, 1, 1))

# img = cv2.imread('images/DIP3E_Original_Images_CH03/Fig0333(a)(test_pattern_blurring_orig).tif')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_gaussian_1 = gaussian_filter(img, 21, 1, 3.5)
# img_gaussian_2 = gaussian_filter(img, 43, 1, 7)

# fig = plt.figure()
# ax1, ax2, ax3 = fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)

# ax1.imshow(img, cmap='gray')
# ax1.set_title('Original Image')
# ax1.axis('off')

# ax2.imshow(img_gaussian_1, cmap='gray')
# ax2.set_title('Kernel of Size: 21, Sigma: 3.5')
# ax2.axis('off')

# ax3.imshow(img_gaussian_2, cmap='gray')
# ax3.set_title('Kernel of Size: 43, Sigma: 7')
# ax3.axis('off')

# plt.show()


# # ORDER-STATISTIC (NONLINEAR) FILTERS
# def k_bubble_sort(nums, kth):
#     for k in range(kth):
#         for i in range(1, len(nums) - k):
#             if nums[i] < nums[i-1]:
#                 temp = nums[i]
#                 nums[i] = nums[i - 1]
#                 nums[i - 1] = temp
    
#     return nums[-kth]

# def median_filter(img, ks):
#     pad = ks // 2
#     img_cp = np.copy(img)
#     img_cp = cv2.copyMakeBorder(img_cp, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
#     img_op = np.copy(img)

#     for x in range(img_cp.shape[0] - 2 * pad):
#         for y in range(img_cp.shape[1] - 2 * pad):
#             img_sl = img_cp[x: x + 2 * pad + 1, y: y + 2 * pad + 1]
#             nums = img_cp.ravel()
            
#             xi = k_bubble_sort(nums, 2 * ks // 2)
#             img_op[x, y] = xi
    
#     return img_op

# img = cv2.imread('images/DIP3E_Original_Images_CH03/Fig0335(a)(ckt_board_saltpep_prob_pt05).tif')
# img = cv2.imread('images/DIP3E_Original_Images_CH03/Fig0333(a)(test_pattern_blurring_orig).tif')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_gaussian = gaussian_filter(img, 3, 1, 19)
# img_md = cv2.medianBlur(img, 7)
# # img_md = median_filter(img, 3)

# fig = plt.figure()
# ax1, ax2, ax3 = fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)

# ax1.imshow(img, cmap='gray')
# ax1.set_title('Original Image')
# ax1.axis('off')

# ax2.imshow(img_gaussian, cmap='gray')
# ax2.set_title('19 X 19 with Sigma: 3')
# ax2.axis('off')

# ax3.imshow(img_md, cmap='gray')
# ax3.set_title('7 X 7')
# ax3.axis('off')
# plt.show()


# ## Laplacian 
# def laplacian(img, ks, ky):

#     pad = ks // 2
#     img_cp = np.copy(img)
#     img_cp = cv2.copyMakeBorder(img_cp, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
#     img_op = np.copy(img_cp)

#     if ky == ks**2 - 1:
#         center_coef = ks**2
#         for x in range(img_cp.shape[0]):
#             for y in range(img_cp.shape[1] -  2 * pad):
#                 img_sl = img_cp[x, y: y + 2 * pad + 1]
#                 C = 0
#                 for n in range(ks):
#                     C += img_sl[n]
#                 img_op[x, y + pad] = C

#         for x in range(img_op.shape[0] - 2 * pad):
#             for y in range(pad, img_op.shape[1] -  pad):
#                 img_sl = img_op[x: x+ 2 * pad + 1, y]
#                 C = 0
#                 for m in range(ks):
#                     C += img_sl[m]
#                 img_cp[x + pad, y] = C

#         img_op = img_cp[pad:-pad, pad:-pad]

#         for x in range(img.shape[0]):
#             for y in range(img.shape[1]):
#                 img_op[x][y] -= center_coef * img[x][y]
        
#         return img_op
    
#     else:
#         center_coef = ky + 2
#         for x in range(pad, img_cp.shape[0] -  pad):
#             for y in range(pad, img_cp.shape[1] -  pad):

#                 C = 0
#                 for row in img_cp[x - pad: x +  pad + 1, y + pad]:
#                     C += row

#                 for col in img_cp[x +  pad, y - pad: y +  pad + 1]:
#                     C += col
                
#                 img_op[x][y] = C - center_coef * img_cp[x][y]
        
#         return img_op[pad:-pad, pad:-pad]


# img = cv2.imread('images/DIP3E_Original_Images_CH03/Fig0338(a)(blurry_moon).tif')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = np.array(img, dtype=int)

# # img = np.array([[8, 8, 8, 1, 6, 6, 6],
# #                 [8, 8, 8, 1, 6, 6, 6],
# #                 [8, 8, 8, 1, 6, 6, 6],
# #                 [3, 3, 3, 5, 7, 7, 7],
# #                 [4, 4, 4, 9, 2, 2, 2],
# #                 [4, 4, 4, 9, 2, 2, 2],
# #                 [4, 4, 4, 9, 2, 2, 2]])

# lp_vh = laplacian(img, 3, 4)
# lp_rvh = laplacian(img, 3, 8)

# C = -1
# img_lp_vh = np.clip(img + C * lp_vh, 0, 255)
# img_lp_rvh = np.clip(img + C * lp_rvh, 0, 255)

# fig = plt.figure()
# ax1, ax2, ax3 = fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)

# ax1.imshow(img, cmap='gray')
# ax1.set_title('Original Image')
# ax1.axis('off')

# ax2.imshow(img_lp_vh, cmap='gray')
# ax2.set_title('4 Kernel Yields')
# ax2.axis('off')

# ax3.imshow(img_lp_rvh, cmap='gray')
# ax3.set_title('8 Kernel Yields')
# ax3.axis('off')

# plt.show()


def unsharp_masking(img, K, ks, sigma):

    img_blur = cv2.GaussianBlur(img, (ks, ks), sigma, sigma)
    g_mask = img - img_blur
    g = img + K * g_mask

    return img_blur, g_mask, g

img = cv2.imread('images/DIP3E_Original_Images_CH03/Fig0340(a)(dipxe_text).tif')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sigma = 2
ks = 15
img_blur, mask, img_um = unsharp_masking(img, 1, ks , sigma)
img_um_ = unsharp_masking(img, 4.5, ks, sigma)[2]

fig = plt.figure()
fig.patch.set_alpha(0.)
ax1, ax2, ax3 = fig.add_subplot(231), fig.add_subplot(232), fig.add_subplot(233)
ax4, ax5 = fig.add_subplot(234), fig.add_subplot(235)

ax1.imshow(img, cmap='gray')
ax1.set_title('Original Image')
ax1.axis('off')

ax2.imshow(img_blur, cmap='gray')
ax2.set_title(f'{ks} x {ks} with sigma: {sigma}')
ax2.axis('off')

ax3.imshow(mask, cmap='gray')
ax3.set_title('Mask')
ax3.axis('off')

ax4.imshow(img_um, cmap='gray')
ax4.set_title('k = 1')
ax4.axis('off')

ax5.imshow(img_um_, cmap='gray')
ax5.set_title('k = 4.5')
ax5.axis('off')


plt.show()



