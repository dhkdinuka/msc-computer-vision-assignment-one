import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("images/highlights_and_shadows.jpg")
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
L, a, b = cv.split(lab)

# Gamma correction
gamma = 0.5  # try 0.5, 1.5
L_norm = L / 255.0
L_corr = np.power(L_norm, gamma) * 255
L_corr = L_corr.astype('uint8')

# Merge back
lab_corr = cv.merge([L_corr, a, b])
img_corr = cv.cvtColor(lab_corr, cv.COLOR_LAB2BGR)

# Histograms
plt.hist(L.ravel(), 256, [0,256], color='blue', alpha=0.5, label="Original L")
plt.hist(L_corr.ravel(), 256, [0,256], color='red', alpha=0.5, label="Corrected L")
plt.legend(), plt.show()

cv.imshow("Gamma Corrected", img_corr)
cv.waitKey(0), cv.destroyAllWindows()
