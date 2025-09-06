import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Gamma
gamma = 0.7  # e.g., 0.70 brightens; pick what looks best and report it

#load the  image
img_bgr = cv.imread('images/highlights_and_shadows.jpg', cv.IMREAD_COLOR)

# ----- convert to Lab and split -----
lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)
L, a, b = cv.split(lab)

# t[i] = ((i/255)^gamma)*255 for i = 0..255
t = np.array([((i/255.0)**gamma)*255 for i in range(256)], dtype=np.uint8)

# apply gamma ONLY on L (same style as: g = t[f])
L_corr = t[L]
# ----- merge back and convert to BGR for display -----
lab_corr = cv.merge([L_corr, a, b])
img_corr_bgr = cv.cvtColor(lab_corr, cv.COLOR_LAB2BGR)

# show images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB));      ax[0].set_title('Original');                 ax[0].axis('off')
ax[1].imshow(cv.cvtColor(img_corr_bgr, cv.COLOR_BGR2RGB)); ax[1].set_title(f'Gamma corrected (γ={gamma})'); ax[1].axis('off')
plt.tight_layout(); plt.show()

# Histograms
hist_L  = cv.calcHist([L],      [0], None, [256], [0,256]).ravel()
hist_Lc = cv.calcHist([L_corr], [0], None, [256], [0,256]).ravel()

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(hist_L);  plt.title('Histogram of L (original)')
plt.xlim([0,255]); plt.xlabel('L intensity'); plt.ylabel('Count'); plt.grid(True, alpha=0.3)

plt.subplot(1,2,2)
plt.plot(hist_Lc); plt.title(f'Histogram of L (γ={gamma})')
plt.xlim([0,255]); plt.xlabel('L intensity'); plt.ylabel('Count'); plt.grid(True, alpha=0.3)

plt.tight_layout(); plt.show()

print(f"Gamma used (γ) = {gamma}")
