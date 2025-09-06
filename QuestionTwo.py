import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#Method to build piecewise_lut
def build_piecewise_lut(points_xy):
    pts = np.array(points_xy, dtype=int)
    if pts[0,0] != 0:   pts = np.vstack(([0, pts[0,1]], pts))
    if pts[-1,0] != 255: pts = np.vstack((pts, [255, pts[-1,1]]))
    pts = pts[np.argsort(pts[:,0])]  # sort by x

    lut = np.zeros(256, dtype='uint8')
    for i in range(len(pts)-1):
        x0,y0 = pts[i]
        x1,y1 = pts[i+1]
        seg = np.linspace(y0, y1, (x1 - x0) + 1)
        lut[x0:x1+1] = np.rint(seg).astype('uint8')
    return lut

#Method to show grey
def show_gray(img, title=""):
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title(title); plt.axis('off')

def plot_transform(lut, title="Transform"):
    plt.plot(np.arange(256), lut)
    plt.title(title)
    plt.xlim([0,255]); plt.ylim([0,255])
    plt.xlabel("Input intensity"); plt.ylabel("Output intensity")
    plt.grid(True, alpha=0.3)

img = cv.imread('images/brain_proton_density_slice.png', cv.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Update path: images/q2_brain_pd.png")

#Accentuate WHITE matter
c_white = [
    (0,   0),
    (60,  25),   # keep background low
    (95,  140),  # strong lift in WM band ~80-110
    (140, 215),  # taper into GM/CSF
    (255, 235)   # compress the brightest CSF a bit
]
lut_white = build_piecewise_lut(c_white)
img_white = cv.LUT(img, lut_white)

#Accentuate GREY matter
c_gray = [
    (0,   0),
    (70,  35),   # compress lows
    (130, 200),  # strong lift in GM band ~120-160
    (190, 235),  # gentle roll-off
    (255, 245)   # keep CSF from washing out
]
lut_gray = build_piecewise_lut(c_gray)
img_graym = cv.LUT(img, lut_gray)

plt.figure(figsize=(12,6))
plt.subplot(2,3,1); show_gray(img, "Original (PD)")
plt.subplot(2,3,2); show_gray(img_white, "Accentuate WHITE matter")
plt.subplot(2,3,3); show_gray(img_graym, "Accentuate GRAY matter")
plt.subplot(2,3,5); plot_transform(lut_white, "Transform for WHITE matter")
plt.subplot(2,3,6); plot_transform(lut_gray,  "Transform for GRAY matter")
plt.tight_layout(); plt.show()
