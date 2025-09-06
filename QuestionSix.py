import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Load image (BGR→Gray)
# -------------------------
IMG_PATH = 'images/einstein.png'  # change if needed
im_bgr = cv.imread(IMG_PATH)
if im_bgr is None:  # fallback to the file you shared
    im_bgr = cv.imread('/mnt/data/cf0e018c-cc10-47aa-8693-e5b0b930a0f3.png')
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

# -------------------------
# 6(a) Use existing filter2D
# -------------------------
gx_a = cv.filter2D(gray, cv.CV_32F, KX)
gy_a = cv.filter2D(gray, cv.CV_32F, KY)
mag_a = cv.magnitude(gx_a, gy_a)

# -------------------------
# 6(b) Write your own Sobel (manual correlation)
# -------------------------
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

# -------------------------
# 6(c) Use separability:
#     [ [1,0,-1],[2,0,-2],[1,0,-1] ] = [1,2,1]^T * [1,0,-1]
#     So Gx: kx = [1,0,-1], ky = [1,2,1]
#     Gy: transpose (kx=[1,2,1], ky=[1,0,-1])
# -------------------------
s = np.array([1, 2, 1], dtype=np.float32)   # smoothing
d = np.array([1, 0, -1], dtype=np.float32)  # derivative

gx_c = cv.sepFilter2D(gray, cv.CV_32F, d, s)  # horiz deriv, vert smooth
gy_c = cv.sepFilter2D(gray, cv.CV_32F, s, d)  # vert deriv, horiz smooth
mag_c = cv.magnitude(gx_c, gy_c)

# -------------------------
# Display (no axis values)
# -------------------------
plt.figure(figsize=(12, 10))

# Row A: filter2D
plt.subplot(3,4,1);  plt.imshow(gray, cmap='gray');   plt.title("Original");             plt.xticks([]); plt.yticks([])
plt.subplot(3,4,2);  plt.imshow(to_u8(gx_a), cmap='gray'); plt.title("A: Gx (filter2D)");  plt.xticks([]); plt.yticks([])
plt.subplot(3,4,3);  plt.imshow(to_u8(gy_a), cmap='gray'); plt.title("A: Gy (filter2D)");  plt.xticks([]); plt.yticks([])
plt.subplot(3,4,4);  plt.imshow(to_u8(mag_a), cmap='gray');plt.title("A: |G|");           plt.xticks([]); plt.yticks([])

# Row B: manual
plt.subplot(3,4,5);  plt.imshow(gray, cmap='gray');   plt.title("Original");             plt.xticks([]); plt.yticks([])
plt.subplot(3,4,6);  plt.imshow(to_u8(gx_b), cmap='gray'); plt.title("B: Gx (manual)");    plt.xticks([]); plt.yticks([])
plt.subplot(3,4,7);  plt.imshow(to_u8(gy_b), cmap='gray'); plt.title("B: Gy (manual)");    plt.xticks([]); plt.yticks([])
plt.subplot(3,4,8);  plt.imshow(to_u8(mag_b), cmap='gray');plt.title("B: |G|");           plt.xticks([]); plt.yticks([])

# Row C: separable
plt.subplot(3,4,9);  plt.imshow(gray, cmap='gray');   plt.title("Original");             plt.xticks([]); plt.yticks([])
plt.subplot(3,4,10); plt.imshow(to_u8(gx_c), cmap='gray'); plt.title("C: Gx (separable)"); plt.xticks([]); plt.yticks([])
plt.subplot(3,4,11); plt.imshow(to_u8(gy_c), cmap='gray'); plt.title("C: Gy (separable)"); plt.xticks([]); plt.yticks([])
plt.subplot(3,4,12); plt.imshow(to_u8(mag_c), cmap='gray');plt.title("C: |G|");           plt.xticks([]); plt.yticks([])

plt.tight_layout()
plt.show()
