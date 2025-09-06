import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# Load image (BGR) and prep RGB
IMG_PATH = "images/daisy.jpg"
img_bgr = cv.imread(IMG_PATH)
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
H, W = img_rgb.shape[:2]


# (a) GrabCut segmentation
mask = np.zeros((H, W), np.uint8)
rect = (W//10, H//10, W - 2*(W//10), H - 2*(H//10))  # x,y,w,h

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

cv.grabCut(img_bgr, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

fg_mask = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 1, 0).astype(np.uint8)

# Optional cleanup for nicer edges
kernel = np.ones((3, 3), np.uint8)
fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, kernel, iterations=1)
fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_CLOSE, kernel, iterations=2)

# Derive foreground and background images
fg_rgb = (img_rgb * fg_mask[..., None]).astype(np.uint8)
bg_rgb = (img_rgb * (1 - fg_mask[..., None])).astype(np.uint8)
# (b) Enhanced image:
blurred_rgb = cv.GaussianBlur(img_rgb, (31, 31), 0)
# Hard matte
enh_hard = (img_rgb * fg_mask[..., None] + blurred_rgb * (1 - fg_mask[..., None])).astype(np.uint8)
# Soft matte (feathered mask) for smoother transition
alpha = cv.GaussianBlur(fg_mask.astype(np.float32), (0, 0), 2.0)
alpha = np.clip(alpha, 0.0, 1.0)[..., None]
enh_soft = (img_rgb.astype(np.float32) * alpha + blurred_rgb.astype(np.float32) * (1 - alpha)).astype(np.uint8)

# ----------------------------
# Display
# ----------------------------
plt.figure(figsize=(14, 10))

# (a) mask, foreground, background
plt.subplot(2, 3, 1); plt.imshow(img_rgb);     plt.title("Original"); plt.xticks([]); plt.yticks([])
plt.subplot(2, 3, 2); plt.imshow(fg_mask*255, cmap="gray"); plt.title("Segmentation Mask"); plt.xticks([]); plt.yticks([])
plt.subplot(2, 3, 3); plt.imshow(fg_rgb);     plt.title("Foreground"); plt.xticks([]); plt.yticks([])
plt.subplot(2, 3, 6); plt.imshow(bg_rgb);     plt.title("Background"); plt.xticks([]); plt.yticks([])

# (b) original vs enhanced (soft-masked)
plt.subplot(2, 3, 4); plt.imshow(img_rgb);    plt.title("Original"); plt.xticks([]); plt.yticks([])
plt.subplot(2, 3, 5); plt.imshow(enh_soft);   plt.title("Enhanced (Blurred BG)"); plt.xticks([]); plt.yticks([])

plt.tight_layout(); plt.show()
