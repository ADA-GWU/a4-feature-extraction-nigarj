import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Load the input image
image_path = 'images/original/book.png'
img = cv2.imread(image_path)
rows, cols, _ = img.shape

# Define source and destination points for perspective warp
src_pts = np.float32([
    [0, 0],
    [cols - 1, 0],
    [cols - 1, rows - 1],
    [0, rows - 1]
])

dst_pts = np.float32([
    [50, 50],
    [cols - 50, 30],
    [cols - 30, rows - 50],
    [30, rows - 30]
])

# Compute perspective transform matrix and apply warp
matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(img, matrix, (cols, rows))

# Create output directories
warped_dir = 'images/warped/'
matches_dir = 'images/output/'
os.makedirs(warped_dir, exist_ok=True)
os.makedirs(matches_dir, exist_ok=True)

# Save the warped image
warped_path = os.path.join(warped_dir, 'warped.jpg')
cv2.imwrite(warped_path, warped)

# ORB feature detection and matching
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img, None)
kp2, des2 = orb.detectAndCompute(warped, None)

# Match features using brute-force with Hamming distance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Compute average distance of top matches
num_matches_to_draw = 50
avg_distance = sum(m.distance for m in matches[:num_matches_to_draw]) / num_matches_to_draw
print("Average match distance:", avg_distance)

# Draw matches
matched_img = cv2.drawMatches(img, kp1, warped, kp2, matches[:num_matches_to_draw], None, flags=2)

# Save and display match visualization
matched_img_path = os.path.join(matches_dir, 'matched_features.jpg')
cv2.imwrite(matched_img_path, matched_img)

# Display only the final match visualization image
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
plt.title("ORB Feature Matching")
plt.axis('off')
plt.show()
