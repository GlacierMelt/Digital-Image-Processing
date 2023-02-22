import numpy as np
import cv2
import matplotlib.pyplot as plt


# Guassian Lowpass Filter
def gaussLowPassFilter(shape, radius=10):  # Gaussian low pass filter
    # Gaussian filter:# Gauss = 1/(2*pi*s2) * exp(-(x**2+y**2)/(2*s2))
    u, v = np.mgrid[-1:1:2.0/shape[0], -1:1:2.0/shape[1]]
    D = np.sqrt(u**2 + v**2)
    D0 = radius / shape[0]
    kernel = np.exp(- (D ** 2) / (2 *D0**2))
    return kernel

def dft2Image(image):  # Optimal extended fast Fourier transform
    # Centralized 2D array f (x, y) * - 1 ^ (x + y)
    mask = np.ones(image.shape)
    mask[1::2, ::2] = -1
    mask[::2, 1::2] = -1
    fImage = image * mask  # f(x,y) * (-1)^(x+y)

    # Optimal DFT expansion size
    rows, cols = image.shape[:2]  # The height and width of the original picture
    rPadded = cv2.getOptimalDFTSize(rows)  # Optimal DFT expansion size
    cPadded = cv2.getOptimalDFTSize(cols)  # For fast Fourier transform

    # Edge extension (complement 0), fast Fourier transform
    dftImage = np.zeros((rPadded, cPadded, 2), np.float32)  # Edge expansion of the original image
    dftImage[:rows, :cols, 0] = fImage  # Edge expansion, 0 on the lower and right sides
    cv2.dft(dftImage, dftImage, cv2.DFT_COMPLEX_OUTPUT)  # fast Fourier transform 
    return dftImage


img = cv2.imread('images/DIP3E_Original_Images_CH04/Fig0441(a)(characters_test_pattern).tif')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

fig = plt.figure()
fig.patch.set_alpha(0.)
ax1, ax2, ax3 = fig.add_subplot(231), fig.add_subplot(232), fig.add_subplot(233)
ax4, ax5, ax6 = fig.add_subplot(234), fig.add_subplot(235), fig.add_subplot(236)

ax1.imshow(img, cmap='gray')
ax1.set_title('Original Image')
ax1.axis('off')


dftImage = dft2Image(img)  # Fast Fourier transform (rPad, cPad, 2)
rPadded, cPadded = dftImage.shape[:2]  # Fast Fourier transform size, original image size optimization

rows, cols = img.shape[:2]
D0 = [10, 30, 60, 90, 120]
for k in range(5):
    # (3) Construct Gaussian low pass filter
    lpFilter = gaussLowPassFilter((rPadded, cPadded), radius=D0[k])

    # (5) Modify Fourier transform in frequency domain: Fourier transform point multiplication low-pass filter
    dftLPfilter = np.zeros(dftImage.shape, dftImage.dtype)  # Size of fast Fourier transform (optimized size)
    for j in range(2):
        dftLPfilter[:rPadded, :cPadded, j] = dftImage[:rPadded, :cPadded, j] * lpFilter

    # (6) The inverse Fourier transform is performed on the low-pass Fourier transform, and only the real part is taken
    idft = np.zeros(dftImage.shape[:2], np.float32)  # Size of fast Fourier transform (optimized size)
    cv2.dft(dftLPfilter, idft, cv2.DFT_REAL_OUTPUT + cv2.DFT_INVERSE + cv2.DFT_SCALE)

    # (7) Centralized 2D array g (x, y) * - 1 ^ (x + y)
    mask2 = np.ones(dftImage.shape[:2])
    mask2[1::2, ::2] = -1
    mask2[::2, 1::2] = -1
    idftCen = idft * mask2  # g(x,y) * (-1)^(x+y)

    # (8) Intercept the upper left corner, the size is equal to the input image
    result = np.clip(idftCen, 0, 255)  # Truncation function, limiting the value to [0, 255]
    imgLPF = result.astype(np.uint8)
    imgLPF = imgLPF[:rows, :cols]

    # (9) Display the result
    eval(f'ax{k+2}').imshow(imgLPF, cmap='gray')
    eval(f'ax{k+2}').set_title(f'D0 = {D0[k]}')
    eval(f'ax{k+2}').axis('off')

plt.show()