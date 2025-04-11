import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# --- Paths Setup ---
input_path = 'images/original/book.png'
warped_dir = 'images/warped/'
matches_dir = 'images/output/'
os.makedirs(warped_dir, exist_ok=True)
os.makedirs(matches_dir, exist_ok=True)

# --- Load Image ---
img = cv2.imread(input_path)
if img is None:
    raise FileNotFoundError(f"Image not found at {input_path}")
rows, cols = img.shape[:2]

# --- Define Perspective Transform ---
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

matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(img, matrix, (cols, rows))

# --- Save Warped Image ---
cv2.imwrite(os.path.join(warped_dir, 'warped.jpg'), warped)

# --- ORB Feature Detection ---
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img, None)
kp2, des2 = orb.detectAndCompute(warped, None)

# --- Feature Matching ---
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# --- Evaluate and Draw Matches ---
num_matches_to_draw = min(50, len(matches))
avg_distance = np.mean([m.distance for m in matches[:num_matches_to_draw]])

matched_img = cv2.drawMatches(img, kp1, warped, kp2, matches[:num_matches_to_draw], None, flags=2)
cv2.imwrite(os.path.join(matches_dir, 'matched_features.jpg'), matched_img)

# --- Save Comparison Image (Optional) ---
comparison = np.hstack((img, warped))
cv2.imwrite(os.path.join(warped_dir, 'original_vs_warped.jpg'), comparison)

# --- Log Average Match Distance ---
with open(os.path.join(matches_dir, 'match_log.txt'), 'w') as f:
    f.write(f"Average match distance (top {num_matches_to_draw}): {avg_distance:.2f}\n")

# --- Visualization ---
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
plt.title(f"ORB Feature Matching (Avg. Distance: {avg_distance:.2f})")
plt.axis('off')
plt.tight_layout()
plt.show()
