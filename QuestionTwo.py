import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


img = cv.imread("images/brain_proton_density_slice.png", cv.IMREAD_GRAYSCALE)

# White matter enhancement
points_input_wm = [0, 120, 200, 255]
points_output_wm = [0, 150, 255, 255]
t_wm = np.interp(np.arange(256), points_input_wm, points_output_wm).astype('uint8')
white_matter = cv.LUT(img, t_wm)

# Gray matter enhancement
points_input_gm = [0, 80, 150, 255]
points_output_gm = [0, 100, 220, 255]
t_gm = np.interp(np.arange(256), points_input_gm, points_output_gm).astype('uint8')
gray_matter = cv.LUT(img, t_gm)

plt.figure(figsize=(12,4))
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title("Original")
plt.subplot(132), plt.imshow(white_matter, cmap='gray'), plt.title("White Matter")
plt.subplot(133), plt.imshow(gray_matter, cmap='gray'), plt.title("Gray Matter")
plt.show()
