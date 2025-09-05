import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv.imread("images/emma.jpg", cv.IMREAD_GRAYSCALE)

# Define input-output mapping
points_input = [0, 50, 150, 151, 255]
points_output = [0, 50, 255, 150, 255]

# Build lookup table
transform = np.interp(np.arange(256), points_input, points_output).astype('uint8')

# Apply LUT
out = cv.LUT(img, transform)

# Display
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title("Original")
plt.subplot(122), plt.imshow(out, cmap='gray'), plt.title("Transformed")
plt.show()
