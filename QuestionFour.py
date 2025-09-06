import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("images/spider.png")
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
h, s, v = cv.split(hsv)

# Apply vibrance formula
sigma = 70
a = 0.6  # adjust
x = np.arange(256)
vibrance_curve = np.minimum(x + a*128*np.exp(-((x-128)**2)/(2*sigma**2)), 255).astype('uint8')
s_new = cv.LUT(s, vibrance_curve)

# Recombine
hsv_new = cv.merge([h, s_new, v])
img_out = cv.cvtColor(hsv_new, cv.COLOR_HSV2BGR)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].set_title('Original Image')
ax[0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

ax[1].set_title(f'Vibrance Enhanced (a={a})')
ax[1].imshow(cv.cvtColor(img_out, cv.COLOR_BGR2RGB))

for a_ax in ax:   # hide ticks & frame
    a_ax.axis('off')

plt.tight_layout()
plt.show()
