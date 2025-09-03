import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Define control points from the graph
# (input, output)
points = [(0, 0), (50, 100), (100, 200), (150, 255), (200, 150), (255, 255)]

# Initialize lookup table
LUT = np.zeros(256, dtype=np.uint8)

# Fill LUT piecewise between control points
for i in range(len(points)-1):
    x1, y1 = points[i]
    x2, y2 = points[i+1]
    slope = (y2 - y1) / (x2 - x1)
    for x in range(x1, x2+1):
        LUT[x] = np.clip(y1 + slope * (x - x1), 0, 255)

# Read grayscale image
img_orig = cv.imread('images/emma.jpg', cv.IMREAD_GRAYSCALE)

# Apply transformation
image_transformed = cv.LUT(img_orig, LUT)

# Show original and transformed images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img_orig, cmap='gray')
ax[0].set_title("Original Image")
ax[0].axis('off')

ax[1].imshow(image_transformed, cmap='gray')
ax[1].set_title("After Intensity Transformation")
ax[1].axis('off')

plt.show()

# Optional: plot transformation curve
plt.plot(LUT, color='blue')
plt.title("Intensity Transformation Curve")
plt.xlabel("Input Intensity")
plt.ylabel("Output Intensity")
plt.show()
