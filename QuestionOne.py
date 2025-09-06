import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load image
img_orig = cv.imread('images/emma.jpg', cv.IMREAD_GRAYSCALE)

#Control Points
c = np.array([(50, 100), (150, 255)])

t1 = np.linspace(0, 50, 50).astype('uint8')              # 50 values -> x = 0..49
# 50..150 -> 100..255 (inclusive)
t2 = np.linspace(c[0,1], c[1,1], (c[1,0] - c[0,0]) + 1).astype('uint8')  # 101 values -> x = 50..150
# 151..255 -> 150..255 (slope 1)
t3 = np.arange(c[1,0] + 1, 256, dtype='uint8')           # 105 values -> x = 151..255, y = x
# Build LUT and enforce the vertical drop at x=150
transform = np.concatenate((t1, t2, t3))
transform[150] = 150

assert transform.size == 256
print(len(transform))
print(transform)
# Apply
image_transformed = cv.LUT(img_orig, transform)

fixe, ax = plt.subplots(1, 2, figsize=(12, 8))
ax[0].set_title('Original Image')
ax[0].imshow(img_orig, cmap='gray')
ax[1].set_title('Transformed Image')
ax[1].imshow(image_transformed, cmap='gray')
for a in ax:
 a.axis('off')
plt.show()
