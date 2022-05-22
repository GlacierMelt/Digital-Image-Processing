import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('images/DIP3E_Original_Images_CH03/Fig0316(4).tif')

row , col = 2, 4
for i in range(col):
    plt.subplot(row, col, i+1)
    img = cv2.imread(f'images/DIP3E_Original_Images_CH03/Fig0316({str(i+1)}).tif')
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(row, col, i+col+1)
    plt.hist(img.ravel(), 256, [0, 256])
    plt.yticks([])

plt.show()


# Histogram Equalization
def hist_equalization(img):

    # Compute the pdf
    pr = np.histogram(img.ravel(), bins=256, \
        range=(0, 255))[0] / img.size 

    op_img = np.copy(img)

    y_points = []
    cdf_i = 0
    # Compute the cdf
    for i in range(256):
        cdf_i += pr[i]
        op_img[img == i] = cdf_i * 255
        y_points.append(cdf_i * 255)

    return op_img, y_points

img = cv2.imread('images/DIP3E_Original_Images_CH03/Fig0316(2).tif')

he_img, y_points = hist_equalization(img)

fig = plt.figure()
ax1, ax2 = fig.add_subplot(231), fig.add_subplot(232)
ax4, ax5, ax6 = fig.add_subplot(234), fig.add_subplot(235), fig.add_subplot(236)


ax1.imshow(img)
ax1.set_title('Original Image')
ax1.axis('off')

ax2.hist(img.ravel(), 256, [0, 256])
ax2.set_xticks([], minor=False)
ax2.set_yticks([], minor=False)

ax4.imshow(he_img)
ax4.set_title('Histogram Equalization')
ax4.axis('off')

ax5.plot(np.arange(256), y_points)
ax5.set_xlabel('r', fontweight = 'bold')
ax5.set_ylabel('s', fontweight = 'bold')
ax5.yaxis.tick_right()
ax5.xaxis.tick_top()

ax6.hist(he_img.ravel(), 256, [0, 256], orientation='horizontal')
ax6.set_xticks([], minor=False)
ax6.set_yticks([], minor=False)

plt.show()


# Histogram Matching (On Hold)
# def hist_matching(img, ref_img):
#     pass



# img = cv2.imread('images/DIP3E_Original_Images_CH03/Fig0323(a)(mars_moon_phobos).tif')
# _, y_points = hist_equalization(img)
# hm_img, y_points = hist_matching(img, img)

# plt.imshow(hm_img)
# plt.show()

# Local Histogram Processing or CLAHE Histogram Equalization（自适应直方图处理）
img = cv2.imread('images/DIP3E_Original_Images_CH03/Fig0326(a)(embedded_square_noisy_512).tif')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def clahe_hist_equalization(img):
    clahe = cv2.createCLAHE(clipLimit=255, tileGridSize=(255, 255))
    return clahe.apply(img)

fig = plt.figure()
ax1, ax2, ax3 = fig.add_subplot(231), fig.add_subplot(232), fig.add_subplot(233)
he_img = hist_equalization(img)[0]
lhp_img = clahe_hist_equalization(img)

ax1.imshow(img, cmap='gray')
ax1.set_title('Original Image')
ax1.axis('off')
ax2.imshow(he_img, cmap='gray')
ax2.set_title('Histogram Equalization')
ax2.axis('off')
ax3.imshow(lhp_img, cmap='gray')
ax3.set_title('CLAHE Histogram Equalization')
ax3.axis('off')

plt.show()
