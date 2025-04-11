import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.filters import gaussian
from skimage.segmentation import active_contour

# --------- Step 1: Load and Resize Image ----------
img_path = 'images/original/bike.jpeg'
img = cv2.imread(img_path)
# Stronger Gaussian blur and histogram equalization
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_eq = cv2.equalizeHist(gray)  # Better contrast
gray_float = img_as_float(gray_eq)
img_resized = gray_float
points = []

# --------- Step 2: Mouse Callback to Draw Points ----------
def draw_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img_resized, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow("Click points (ESC when done)", img_resized)

cv2.imshow("Click points (ESC when done)", img_resized)
cv2.setMouseCallback("Click points (ESC when done)", draw_points)

print("👉 Click around the object to draw a border, then press ESC...")
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(points) < 3:
    print("❌ Not enough points! Please select at least 3.")
    exit()

# --------- Step 3: Convert & Run Active Contour ----------
init = np.array(points, dtype=np.float32)

gray_float = img_as_float(gray)
blurred = gaussian(gray_float, sigma=2)


snake = active_contour(
    blurred,
    init,
    alpha=0.01,     # smaller alpha -> more flexible
    beta=1,         # smaller beta -> less smoothing
    gamma=0.1,      # larger gamma = faster convergence
    w_edge=1.0,
    w_line=0.0
)

# --------- Step 4: Plot the Results ----------
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(gray, cmap='gray')
ax.plot(init[:, 0], init[:, 1], '--r', label='Initial points')
ax.plot(snake[:, 0], snake[:, 1], '-b', label='Active contour')
ax.set_title('Active Contour Result')
ax.axis('off')
ax.legend()
plt.tight_layout()
plt.savefig('images/')
plt.show()
