import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Core: image zoom (manual)
# ---------------------------
def zoom(img, s: float, method: str = "nearest"):
    """
    Zoom by factor s (0 < s <= 10). Supports 'nearest' and 'bilinear'.
    Works for grayscale or color (H,W[,C]).
    """
    assert s > 0 and s <= 10, "s must be in (0,10]"
    src = img.astype(np.float32)
    H, W = src.shape[:2]
    C = 1 if src.ndim == 2 else src.shape[2]
    H2, W2 = max(1, int(round(H * s))), max(1, int(round(W * s)))

    # map target pixel centers -> source coords
    yy2 = np.arange(H2, dtype=np.float32)
    xx2 = np.arange(W2, dtype=np.float32)
    y = (yy2 + 0.5) / s - 0.5   # shape (H2,)
    x = (xx2 + 0.5) / s - 0.5   # shape (W2,)

    # NEAREST
    if method.lower().startswith("near"):
        yi = np.clip(np.round(y).astype(int), 0, H - 1)
        xi = np.clip(np.round(x).astype(int), 0, W - 1)
        out = src[yi[:, None], xi[None, :]]
        return np.clip(out, 0, 255).astype(np.uint8)

    # BILINEAR
    y0 = np.floor(y).astype(int);  x0 = np.floor(x).astype(int)
    y1 = np.clip(y0 + 1, 0, H - 1); x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1);     x0 = np.clip(x0, 0, W - 1)
    wy = (y - y0).astype(np.float32)[:, None, None]  # (H2,1,1)
    wx = (x - x0).astype(np.float32)[None, :, None]  # (1,W2,1)

    I00 = src[y0[:, None], x0[None, :]]  # top-left
    I01 = src[y0[:, None], x1[None, :]]  # top-right
    I10 = src[y1[:, None], x0[None, :]]  # bottom-left
    I11 = src[y1[:, None], x1[None, :]]  # bottom-right

    # ensure three dims for broadcasting
    if src.ndim == 2:
        I00 = I00[..., None]; I01 = I01[..., None]
        I10 = I10[..., None]; I11 = I11[..., None]

    top = I00 * (1 - wx) + I01 * wx
    bot = I10 * (1 - wx) + I11 * wx
    out = top * (1 - wy) + bot * wy
    out = np.clip(out, 0, 255)

    return out[..., 0].astype(np.uint8) if C == 1 else out.astype(np.uint8)

# ---------------------------------
# Metric: normalized SSD in [0, 1]
# ---------------------------------
def normalized_ssd(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    return float(np.sum((a - b) ** 2) / (a.size * (255.0 ** 2)))

# ---------------------------
# Demo / Test harness (s=4)
# ---------------------------
# Replace these paths with your images (two originals & their zoomed-out versions if provided)
paths = [
    ("images/zoom/im01.png", None),         # will auto-make a small 1/4 version if None
    ("images/zoom/im02.png",  None),
]

fig, axes = plt.subplots(len(paths), 4, figsize=(14, 7), squeeze=False)
for r, (orig_p, small_p) in enumerate(paths):
    # read original as grayscale
    orig = cv.imread(orig_p, cv.IMREAD_GRAYSCALE)
    if orig is None:
        # fallbacks to files you shared in this thread
        fallback = ["/mnt/data/cf0e018c-cc10-47aa-8693-e5b0b930a0f3.png",
                    "/mnt/data/effbc0d2-cdfe-4fa5-96be-29fae81fbcdd.png"]
        orig = cv.imread(fallback[r % len(fallback)], cv.IMREAD_GRAYSCALE)

    # read provided small image, else synthesize by 1/4 downscale (INTER_AREA is best for shrink)
    if small_p is not None:
        small = cv.imread(small_p, cv.IMREAD_GRAYSCALE)
    else:
        small = cv.resize(orig, dsize=None, fx=0.25, fy=0.25, interpolation=cv.INTER_AREA)

    # scale factor to match original size (should be ~4.0)
    s_h = orig.shape[0] / small.shape[0]
    s_w = orig.shape[1] / small.shape[1]
    assert abs(s_h - s_w) < 1e-3, "Non-uniform pair; expected a uniform zoom factor"
    s = (s_h + s_w) / 2.0

    up_nn  = zoom(small, s, method="nearest")
    up_bil = zoom(small, s, method="bilinear")

    nssd_nn  = normalized_ssd(orig, up_nn)
    nssd_bil = normalized_ssd(orig, up_bil)

    # --- Plot (no axis values) ---
    axes[r, 0].imshow(orig, cmap="gray");  axes[r, 0].set_title("Original")
    axes[r, 1].imshow(small, cmap="gray"); axes[r, 1].set_title(f"Small (1/{int(round(s))})")
    axes[r, 2].imshow(up_nn, cmap="gray"); axes[r, 2].set_title(f"NN up×{s:.1f}\nNSSD={nssd_nn:.4f}")
    axes[r, 3].imshow(up_bil, cmap="gray");axes[r, 3].set_title(f"Bilinear up×{s:.1f}\nNSSD={nssd_bil:.4f}")
    for c in range(4):
        axes[r, c].set_xticks([]); axes[r, c].set_yticks([])

plt.tight_layout(); plt.show()
