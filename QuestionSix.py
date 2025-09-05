import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


img = cv.imread("images/einstein.png", cv.IMREAD_GRAYSCALE)

# (a) Using filter2D
sobel_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float32)
sobel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.float32)
gx = cv.filter2D(img, -1, sobel_x)
gy = cv.filter2D(img, -1, sobel_y)
grad = cv.magnitude(gx.astype(np.float32), gy.astype(np.float32))

# (b) Manual convolution
def conv2d(img, kernel):
    k = kernel.shape[0]//2
    out = np.zeros_like(img)
    for i in range(k, img.shape[0]-k):
        for j in range(k, img.shape[1]-k):
            out[i,j] = np.sum(img[i-k:i+k+1, j-k:j+k+1] * kernel)
    return out

gx2 = conv2d(img, sobel_x)
gy2 = conv2d(img, sobel_y)
grad2 = cv.magnitude(gx2.astype(np.float32), gy2.astype(np.float32))

plt.subplot(121), plt.imshow(grad, cmap='gray'), plt.title("filter2D")
plt.subplot(122), plt.imshow(grad2, cmap='gray'), plt.title("Manual")
plt.show()
