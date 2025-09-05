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

plt.subplot(121), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)), plt.title("Original")
plt.subplot(122), plt.imshow(cv.cvtColor(img_out, cv.COLOR_BGR2RGB)), plt.title("Vibrance Enhanced")
plt.show()
