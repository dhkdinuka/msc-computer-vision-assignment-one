import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Paths (update to your images)
# ----------------------------
RICE_A = "images/rice_salt_pepper_noise.png"   # salt & pepper like
RICE_B = "images/rice_gaussian_noise.png"   # Gaussian noise like

# ----------------------------
# Helpers
# ----------------------------
def denoise(img, variant="a"):
    """Denoise grayscale image depending on variant."""
    if variant.lower() == "a":
        # salt & pepper -> median filter is best
        return cv.medianBlur(img, 3)
    else:
        # Gaussian -> Gaussian blur (or fastNlMeansDenoising if you prefer)
        return cv.GaussianBlur(img, (5, 5), 0)

def fill_holes(binary):
    """Fill holes inside foreground using flood-fill."""
    h, w = binary.shape
    inv = 255 - binary
    flood = inv.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv.floodFill(flood, mask, (0, 0), 255)
    holes = 255 - flood                # holes become white
    return cv.bitwise_or(binary, holes)

def clean_mask(binary):
    """Remove small specks and fill small gaps/holes."""
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    opened  = cv.morphologyEx(binary, cv.MORPH_OPEN,  k, iterations=1)   # remove tiny dots
    closed  = cv.morphologyEx(opened, cv.MORPH_CLOSE, k, iterations=1)   # bridge small gaps
    filled  = fill_holes(closed)
    return filled

def segment_otsu(img):
    """Otsu segmentation (foreground bright -> white)."""
    # Otsu on denoised image
    thr, bw = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # Ensure rice grains are white; if not, invert
    if np.sum(bw == 255) < bw.size / 2:
        bw = cv.bitwise_not(bw)
    return bw

def count_components(mask, min_area=50):
    """Connected components and area filtering; returns count and labeled image."""
    num, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    # remove background (label 0)
    areas = stats[1:, cv.CC_STAT_AREA]
    keep = np.where(areas >= min_area)[0] + 1  # component labels to keep
    kept_mask = np.isin(labels, keep).astype(np.uint8) * 255
    # make a colored label image for visualization
    lab_vis = (labels.astype(np.float32) / max(1, num - 1) * 255).astype(np.uint8)
    lab_vis = cv.applyColorMap(lab_vis, cv.COLORMAP_JET)
    lab_vis = cv.bitwise_and(lab_vis, lab_vis, mask=kept_mask)
    return len(keep), kept_mask, lab_vis

def process(path, variant):
    """Complete pipeline for a single image."""
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    assert img is not None, f"Could not read {path}"
    den = denoise(img, variant)
    bw  = segment_otsu(den)
    msk = clean_mask(bw)
    # choose area threshold relative to image size (robust across datasets)
    min_area = max(50, int(0.0005 * img.size))   # tune if needed
    n, kept_mask, lab_vis = count_components(msk, min_area=min_area)
    return img, den, bw, kept_mask, lab_vis, n

# ----------------------------
# Run both (a) and (b)
# ----------------------------
imgA, denA, bwA, maskA, labA, nA = process(RICE_A, "a")
imgB, denB, bwB, maskB, labB, nB = process(RICE_B, "b")

# ----------------------------
# Display (no axis values)
# ----------------------------
fig, ax = plt.subplots(2, 5, figsize=(16, 7))
titles = [
    ["(a) Original", "(a) Denoised", "(a) Otsu", "(a) Cleaned mask", f"(a) Labeled (count={nA})"],
    ["(b) Original", "(b) Denoised", "(b) Otsu", "(b) Cleaned mask", f"(b) Labeled (count={nB})"],
]
rows = [(imgA, denA, bwA, maskA, labA), (imgB, denB, bwB, maskB, labB)]

for r in range(2):
    for c in range(5):
        if c < 4:
            ax[r, c].imshow(rows[r][c], cmap="gray")
        else:
            ax[r, c].imshow(rows[r][c])  # color
        ax[r, c].set_title(titles[r][c])
        ax[r, c].set_xticks([]); ax[r, c].set_yticks([])

plt.tight_layout(); plt.show()
