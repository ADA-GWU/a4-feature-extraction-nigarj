import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# --- Constants ---
INPUT_PATH = 'images/original/dog.jpg'
OUTPUT_BASE = 'images/output/harris_corners/'

BLOCK_SIZE = 2
K_VALUES = [0.2]
KERNEL_SIZES = [5, 15, 25]


def ensure_output_dirs(base_path, subfolders):
    """Ensure subfolders for each ksize exist."""
    for folder in subfolders:
        os.makedirs(os.path.join(base_path, f'ksize_{folder}'), exist_ok=True)


def load_and_preprocess_image(path):
    """Load and return color, grayscale and blurred versions of an image."""
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return img, blurred


def detect_and_draw_harris(img, gray_blur, block_size, ksize, k):
    """Perform Harris corner detection and return the corner-marked image."""
    gray_float = np.float32(gray_blur)
    corners = cv2.cornerHarris(gray_float, block_size, ksize, k)
    corners = cv2.dilate(corners, None)

    result = img.copy()
    result[corners > 0.01 * corners.max()] = [0, 255, 0]
    return result


def main():
    ensure_output_dirs(OUTPUT_BASE, KERNEL_SIZES)
    img, blurred = load_and_preprocess_image(INPUT_PATH)

    fig, axes = plt.subplots(1, len(KERNEL_SIZES), figsize=(18, 6))

    for i, ksize in enumerate(KERNEL_SIZES):
        for k in K_VALUES:
            result = detect_and_draw_harris(img, blurred, BLOCK_SIZE, ksize, k)
            output_path = os.path.join(OUTPUT_BASE, f'ksize_{ksize}', 'harris_result.jpg')
            cv2.imwrite(output_path, result)

            axes[i].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            axes[i].set_title(f'Harris Corners\nksize={ksize}')
            axes[i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
