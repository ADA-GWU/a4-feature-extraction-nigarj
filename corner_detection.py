import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# --- Input Setup ---
INPUT_PATH = 'images/original/dog.jpg'
img = cv2.imread(INPUT_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# --- Harris Corner Parameters ---
block_size = 2
k_values = [0.2]  # You can add more `k` values if needed
ksizes = [5, 15, 25]

# --- Harris Corner Detection and Saving ---
output_base = 'images/harris_corners/'

fig, axes = plt.subplots(1, len(ksizes), figsize=(18, 6))

for i, ksize in enumerate(ksizes):
    for k in k_values:
        # Convert image to float32 for Harris
        gray_float = np.float32(blurred)
        harris_corners = cv2.cornerHarris(gray_float, block_size, ksize, k)
        harris_corners = cv2.dilate(harris_corners, None)

        # Copy and mark corners
        corner_img = img.copy()
        corner_img[harris_corners > 0.01 * harris_corners.max()] = [0, 255, 0]  # black corners

        # Save result
        output_path = os.path.join(output_base, f'ksize_{ksize}')
        os.makedirs(output_path, exist_ok=True)
        cv2.imwrite(os.path.join(output_path, 'harris_result.jpg'), corner_img)

        # Display
        axes[i].imshow(cv2.cvtColor(corner_img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f'Harris Corners\nksize={ksize}')
        axes[i].axis('off')

plt.tight_layout()
plt.show()
