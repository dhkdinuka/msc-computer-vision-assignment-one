import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# Inputs
IMG_PATH = "images/sapphire.jpg"   # <- set your Fig. 9 image
img_bgr = cv.imread(IMG_PATH)
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
gray    = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

# (a) Segmentation (Otsu on blurred gray)
blur = cv.GaussianBlur(gray, (5,5), 0)
_, mask = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Sapphires are usually darker than the table → make them white in the mask
if gray[mask==255].mean() > gray[mask==0].mean():
    mask = cv.bitwise_not(mask)
# (b) Morphology: fill holes / refine edges
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


# (c) Connected components -> pixel areas
num, labels, stats, centroids = cv.connectedComponentsWithStats(mask_filled, connectivity=8)
# Skip label 0 (background)
areas_px = stats[1:, cv.CC_STAT_AREA]              # array of size 2 (two sapphires)

# Pretty label visualization
label_img = (labels.astype(np.float32) / max(1, num-1) * 255).astype(np.uint8)
label_vis = cv.applyColorMap(label_img, cv.COLORMAP_JET)
label_vis = cv.bitwise_and(label_vis, label_vis, mask=mask_filled)


# (d) Actual area from pixel area
f_mm             = 8.0            # given
Z_mm             = 480.0          # given (camera height above table)
PIXEL_PITCH_MM   = 0.0014         # <-- set your sensor pixel size (e.g., 1.4 µm = 0.0014 mm)

mm_per_pixel = (Z_mm / f_mm) * PIXEL_PITCH_MM
areas_mm2    = areas_px * (mm_per_pixel ** 2)

print("Pixel areas:", areas_px.tolist())
print("mm^2 areas (using pixel pitch =", PIXEL_PITCH_MM, "mm):", areas_mm2.tolist())
print("Total area (mm^2):", float(np.sum(areas_mm2)))

# -------------------------------
# Display (no axis values) — fixed headers
# -------------------------------
fig, axs = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

views = [
    (img_rgb, None, "Original"),
    (mask, 'gray', "Otsu mask"),
    (mask_filled, 'gray', "Filled mask"),
    (cv.cvtColor(label_vis, cv.COLOR_BGR2RGB), None, "Connected components"),
]

for ax, (im, cmap, title) in zip(axs.ravel(), views):
    ax.imshow(im, cmap=cmap)
    ax.set_title(title, pad=6)
    ax.axis("off")  # hides ticks & spines

plt.show()