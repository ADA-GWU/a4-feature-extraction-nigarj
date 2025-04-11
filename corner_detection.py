import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# --- Input/Output Setup ---
INPUT_PATH = 'images/original/jasmine.jpeg'



# --- Load Image and Convert to Grayscale ---
img = cv2.imread(INPUT_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- 1. Gaussian Blur (Optional Denoising) ---
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# --- 5. Harris Corner Detection ---

# Parameters
block_size = 2      # It is the size of neighbourhood considered for corner detection
ksize = 5           # Aperture parameter for the Sobel operator.
k = 0.2      # Harris detector free parameter

# Convert image to float32 for Harris
gray_float = np.float32(blurred)

# Apply Harris Corner Detection
harris_corners = cv2.cornerHarris(gray_float, block_size, ksize, k)

# Dilate to mark the corners more clearly
harris_corners = cv2.dilate(harris_corners, None)

# Threshold and mark corners in red
corner_img = img.copy()
corner_img[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]  # Red corners

# Save the result
output_path = 'images/harris_corners/'
os.makedirs(output_path, exist_ok=True)
cv2.imwrite(output_path + 'harris_result.jpg', corner_img)

# --- Optional Visualization ---
def show(title, img):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()



# Optional display
show("Harris Corners", corner_img)
