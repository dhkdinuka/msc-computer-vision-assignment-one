import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# Load image (BGR→Gray)
IMG_PATH = 'images/einstein.png'
im_bgr = cv.imread(IMG_PATH)
gray = cv.cvtColor(im_bgr, cv.COLOR_BGR2GRAY)

# Sobel 3x3 kernels
KX = np.array([[1, 0, -1],
               [2, 0, -2],
               [1, 0, -1]], dtype=np.float32)   # horizontal derivative
KY = KX.T                                       # vertical derivative

def to_u8(a):
    a = np.abs(a).astype(np.float32)
    a = cv.normalize(a, None, 0, 255, cv.NORM_MINMAX)
    return a.astype(np.uint8)

# 6(a) Use existing filter2D
gx_a = cv.filter2D(gray, cv.CV_32F, KX)
gy_a = cv.filter2D(gray, cv.CV_32F, KY)
mag_a = cv.magnitude(gx_a, gy_a)

# 6(b) Write your own Sobel (manual correlation)
def correlate2d(img, k):
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    pad = np.pad(img.astype(np.float32), ((ph, ph), (pw, pw)), mode='reflect')
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(pad[i:i+kh, j:j+kw] * k)
    return out

gx_b = correlate2d(gray, KX)
gy_b = correlate2d(gray, KY)
mag_b = cv.magnitude(gx_b, gy_b)

# 6(c) Use separability:
s = np.array([1, 2, 1], dtype=np.float32)   # smoothing
d = np.array([1, 0, -1], dtype=np.float32)  # derivative

gx_c = cv.sepFilter2D(gray, cv.CV_32F, d, s)  # horiz deriv, vert smooth
gy_c = cv.sepFilter2D(gray, cv.CV_32F, s, d)  # vert deriv, horiz smooth
mag_c = cv.magnitude(gx_c, gy_c)


# Display with visible headers (no axis values)

fig, axs = plt.subplots(3, 4, figsize=(14, 9), constrained_layout=True)

# Column headers
col_titles = ["Original", "Gx", "Gy", "|G|"]
for j, t in enumerate(col_titles):
    axs[0, j].set_title(t, pad=6)

# Row headers (labels at left of each row)
row_titles = ["A: filter2D", "B: manual", "C: separable"]
for i, t in enumerate(row_titles):
    axs[i, 0].set_ylabel(t, rotation=0, labelpad=35, va='center', ha='right', fontsize=11)

# Row A
axs[0,0].imshow(gray, cmap='gray')
axs[0,1].imshow(to_u8(gx_a), cmap='gray')
axs[0,2].imshow(to_u8(gy_a), cmap='gray')
axs[0,3].imshow(to_u8(mag_a), cmap='gray')

# Row B
axs[1,0].imshow(gray, cmap='gray')
axs[1,1].imshow(to_u8(gx_b), cmap='gray')
axs[1,2].imshow(to_u8(gy_b), cmap='gray')
axs[1,3].imshow(to_u8(mag_b), cmap='gray')

# Row C
axs[2,0].imshow(gray, cmap='gray')
axs[2,1].imshow(to_u8(gx_c), cmap='gray')
axs[2,2].imshow(to_u8(gy_c), cmap='gray')
axs[2,3].imshow(to_u8(mag_c), cmap='gray')

# Hide ticks/axis values
for ax in axs.ravel():
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

plt.show()


