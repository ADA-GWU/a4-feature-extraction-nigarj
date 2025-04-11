import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# --- Constants ---
INPUT_PATH = 'images/original/apple.jpeg'
OUTPUT_BASE = 'images/output/'
OUTPUT_DIR = {
    "sobel": os.path.join(OUTPUT_BASE, "edge_sobel/"),
    "canny": os.path.join(OUTPUT_BASE, "edge_canny/"),
    "laplacian": os.path.join(OUTPUT_BASE, "edge_laplacian/")
}


def ensure_directories(directories):
    """Create directories if they don't already exist."""
    for path in directories.values():
        os.makedirs(path, exist_ok=True)


def load_and_preprocess_image(path):
    """Load an image and return both grayscale and blurred versions."""
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return img, gray, blurred


def detect_edges(blurred_img):
    """Perform Sobel, Laplacian, and Canny edge detection."""
    # Sobel
    sobelx = cv2.Sobel(blurred_img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred_img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)

    # Laplacian
    laplacian = cv2.Laplacian(blurred_img, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))

    # Canny
    canny_edges = cv2.Canny(blurred_img, 50, 150)

    return sobel_combined, laplacian, canny_edges


def save_results(results, output_dirs):
    """Save edge detection results to corresponding folders."""
    cv2.imwrite(os.path.join(output_dirs["sobel"], 'sobel_edges.jpg'), results["sobel"])
    cv2.imwrite(os.path.join(output_dirs["laplacian"], 'laplacian_edges.jpg'), results["laplacian"])
    cv2.imwrite(os.path.join(output_dirs["canny"], 'canny_edges.jpg'), results["canny"])


def plot_results(gray, blurred, edges):
    """Display grayscale, blurred, and edge-detected images using Matplotlib."""
    titles = ["Grayscale Image", "Gaussian Blurred", "Sobel Edge Detection",
              "Canny Edge Detection", "Laplacian Edge Detection"]
    images = [gray, blurred, edges["sobel"], edges["canny"], edges["laplacian"]]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    ensure_directories(OUTPUT_DIR)

    _, gray, blurred = load_and_preprocess_image(INPUT_PATH)
    sobel, laplacian, canny = detect_edges(blurred)

    edges = {
        "sobel": sobel,
        "laplacian": laplacian,
        "canny": canny
    }

    save_results(edges, OUTPUT_DIR)
    plot_results(gray, blurred, edges)


if __name__ == "__main__":
    main()
