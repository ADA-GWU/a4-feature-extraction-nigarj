import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images
img1 = cv2.imread('image1.png', cv2.IMREAD_GRAYSCALE)  # query image
img2 = cv2.imread('image2.png', cv2.IMREAD_GRAYSCALE)  # train image

# Initialize ORB detector
orb = cv2.ORB_create(nfeatures=1000)

# Find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Create a brute force matcher with Hamming distance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(des1, des2)

# Sort matches by distance (lower is better)
matches = sorted(matches, key=lambda x: x.distance)

# Draw top N matches
num_matches_to_draw = 50
matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:num_matches_to_draw], None, flags=2)

# Plot the matches
plt.figure(figsize=(15, 7))
plt.title('ORB Feature Matching')
plt.imshow(matched_img)
plt.axis('off')
plt.show()

# Match Quality Evaluation: Average Distance
avg_distance = sum([match.distance for match in matches[:num_matches_to_draw]]) / len(matches[:num_matches_to_draw])
print("Average match distance:", avg_distance)
