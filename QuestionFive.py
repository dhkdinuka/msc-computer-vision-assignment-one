import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("images/jeniffer.jpg")
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
h, s, v = cv.split(hsv)

# Threshold to extract mask
_, mask = cv.threshold(v, 120, 255, cv.THRESH_BINARY)

# Foreground
fg = cv.bitwise_and(v, v, mask=mask)

# Histogram equalization using OpenCV
fg_eq = cv.equalizeHist(fg)

# Replace only foreground
v_new = cv.add(cv.bitwise_and(v, v, mask=cv.bitwise_not(mask)), fg_eq)

# Merge back
out = cv.cvtColor(cv.merge([h, s, v_new]), cv.COLOR_HSV2BGR)

plt.subplot(131), plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)), plt.title("Original")
plt.subplot(132), plt.imshow(mask, cmap='gray'), plt.title("Mask")
plt.subplot(133), plt.imshow(cv.cvtColor(out, cv.COLOR_BGR2RGB)), plt.title("Foreground Equalized")
plt.show()
