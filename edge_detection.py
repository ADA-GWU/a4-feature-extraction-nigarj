import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# --- Input/Output Setup ---
INPUT_PATH = 'images/original/apple.jpeg'
OUTPUT_DIR = {
    "sobel": "images/edge_sobel/",
    "canny": "images/edge_canny/",
    "laplacian": "images/edge_laplacian/"
}

# Create output folders if they don't exist
for path in OUTPUT_DIR.values():
    os.makedirs(path, exist_ok=True)

# --- Load Image and Convert to Grayscale ---
img = cv2.imread(INPUT_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- 1. Gaussian Blur (Optional Denoising) ---
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# --- 2. Sobel Edge Detection ---
sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobelx, sobely)

cv2.imwrite(OUTPUT_DIR["sobel"] + 'sobel_edges.jpg', sobel_combined)

# --- 3. Laplacian Edge Detection ---
laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))
cv2.imwrite(OUTPUT_DIR["laplacian"] + 'laplacian_edges.jpg', laplacian)

# --- 4. Canny Edge Detection ---
edges_canny = cv2.Canny(blurred, 50, 150)
cv2.imwrite(OUTPUT_DIR["canny"] + 'canny_edges.jpg', edges_canny)


# Display the results
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
axes[0].imshow(gray, cmap="gray")
axes[0].set_title("Grayscale Image")
axes[0].axis("off")

axes[1].imshow(blurred, cmap="gray")
axes[1].set_title("Gaussian Blurred")
axes[1].axis("off")

axes[2].imshow(sobel_combined, cmap="gray")
axes[2].set_title("Sobel Edge Detection")
axes[2].axis("off")

axes[3].imshow(edges_canny, cmap="gray")
axes[3].set_title("Canny Edge Detection")
axes[3].axis("off")

axes[4].imshow(laplacian, cmap="gray")
axes[4].set_title("Laplacian Edge Detection")
axes[4].axis("off")

plt.show()
