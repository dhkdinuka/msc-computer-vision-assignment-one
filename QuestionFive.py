import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# (a) Open image and split to HSV (show H,S,V later)
img = cv.imread('images/jeniffer.jpg', cv.IMREAD_COLOR)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
h, s, v = cv.split(hsv)          # uint8, 0..255

# (b) Select plane & make a binary foreground mask (Otsu on V)
_, mask = cv.threshold(v, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# (optional) small cleanup
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
mask = cv.morphologyEx(mask, cv.MORPH_OPEN,  kernel, iterations=1)
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=1)

# Use the Value plane as grayscale input f (like the slide)
f = v.copy()

# (c) Foreground only (we’ll use mask in the histogram)
# (d) Histogram and CDF (masked/foreground only)
L = 256
hist_fg = cv.calcHist([f], [0], mask, [L], [0, L]).ravel().astype(np.float64)
cdf_fg  = np.cumsum(hist_fg)

# Foreground pixel count N_fg replaces M*N from the slide
N_fg = int(mask.sum() // 255) if mask is not None else f.size

# (e) Slide formula LUT: t[k] = ((L-1)/(N_fg)) * CDF_fg[k]
t = np.array([ ((L-1) / max(N_fg, 1)) * cdf_fg[k] for k in range(L) ], dtype=np.uint8)

# Apply like on the slide: g = t[f]
g = t[f]  # equalized V values for all pixels

# (f) Replace only the foreground; keep background unchanged
fg_eq = cv.bitwise_and(g, g, mask=mask)
bg    = cv.bitwise_and(f, f, mask=cv.bitwise_not(mask))
v_out = cv.add(bg, fg_eq)

# Recombine and convert back to BGR
out = cv.cvtColor(cv.merge([h, s, v_out]), cv.COLOR_HSV2BGR)

# ----- Display: H, S, V, Mask, Original, Result (axes off) -----
fig, ax = plt.subplots(2, 3, figsize=(12, 8))
ax[0,0].set_title('Hue');        ax[0,0].imshow(h, cmap='gray', vmin=0, vmax=255)
ax[0,1].set_title('Saturation'); ax[0,1].imshow(s, cmap='gray', vmin=0, vmax=255)
ax[0,2].set_title('Value');      ax[0,2].imshow(f, cmap='gray', vmin=0, vmax=255)
ax[1,0].set_title('Mask');       ax[1,0].imshow(mask, cmap='gray', vmin=0, vmax=255)
ax[1,1].set_title('Original');   ax[1,1].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
ax[1,2].set_title('FG equalized'); ax[1,2].imshow(cv.cvtColor(out, cv.COLOR_BGR2RGB))
for a in ax.ravel():
    a.axis('off')
plt.tight_layout()
plt.show()
