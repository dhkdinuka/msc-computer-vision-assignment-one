import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Inputs (update the image path)
# -------------------------------
IMG_PATH = "images/sapphire.jpg"   # <- set your Fig. 9 image
img_bgr = cv.imread(IMG_PATH)
assert img_bgr is not None, "Image not found. Update IMG_PATH."
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
gray    = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

# -------------------------------
# (a) Segmentation (Otsu on blurred gray)
# -------------------------------
blur = cv.GaussianBlur(gray, (5,5), 0)
_, mask = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Sapphires are usually darker than the table → make them white in the mask
if gray[mask==255].mean() > gray[mask==0].mean():
    mask = cv.bitwise_not(mask)

# -------------------------------
# (b) Morphology: fill holes / refine edges
# -------------------------------
k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k, iterations=2)

# Fill tiny pinholes robustly with flood-fill
def fill_holes(bin_img):
    h, w = bin_img.shape
    ff   = cv.bitwise_not(bin_img.copy())
    cv.floodFill(ff, np.zeros((h+2, w+2), np.uint8), (0,0), 255)
    holes = cv.bitwise_not(ff)
    return cv.bitwise_or(bin_img, holes)

mask_filled = fill_holes(mask)

# -------------------------------
# (c) Connected components -> pixel areas
# -------------------------------
num, labels, stats, centroids = cv.connectedComponentsWithStats(mask_filled, connectivity=8)
# Skip label 0 (background)
areas_px = stats[1:, cv.CC_STAT_AREA]              # array of size 2 (two sapphires)

# Pretty label visualization
label_img = (labels.astype(np.float32) / max(1, num-1) * 255).astype(np.uint8)
label_vis = cv.applyColorMap(label_img, cv.COLORMAP_JET)
label_vis = cv.bitwise_and(label_vis, label_vis, mask=mask_filled)

# -------------------------------
# (d) Actual area from pixel area
# Pinhole model: x_world = (Z/f) * x_image
#   ⇒ mm_per_pixel = (Z_mm / f_mm) * pixel_pitch_mm
#   ⇒ area_mm2 = area_px * (mm_per_pixel)^2
# -------------------------------
f_mm             = 8.0            # given
Z_mm             = 480.0          # given (camera height above table)
PIXEL_PITCH_MM   = 0.0014         # <-- set your sensor pixel size (e.g., 1.4 µm = 0.0014 mm)

mm_per_pixel = (Z_mm / f_mm) * PIXEL_PITCH_MM
areas_mm2    = areas_px * (mm_per_pixel ** 2)

print("Pixel areas:", areas_px.tolist())
print("mm^2 areas (using pixel pitch =", PIXEL_PITCH_MM, "mm):", areas_mm2.tolist())
print("Total area (mm^2):", float(np.sum(areas_mm2)))

# -------------------------------
# Display (no axis values)
# -------------------------------
plt.figure(figsize=(12,8))
plt.subplot(2,2,1); plt.imshow(img_rgb);        plt.title("Original");      plt.xticks([]); plt.yticks([])
plt.subplot(2,2,2); plt.imshow(mask, cmap='gray');       plt.title("Otsu mask");     plt.xticks([]); plt.yticks([])
plt.subplot(2,2,3); plt.imshow(mask_filled, cmap='gray');plt.title("Filled mask");   plt.xticks([]); plt.yticks([])
plt.subplot(2,2,4); plt.imshow(label_vis[..., ::-1]);    plt.title("Connected components"); plt.xticks([]); plt.yticks([])
plt.tight_layout(); plt.show()
