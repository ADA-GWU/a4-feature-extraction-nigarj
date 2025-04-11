import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# --- Constants ---
INPUT_PATH = 'images/original/bike.jpeg'
OUTPUT_DIR_LINES = 'images/output/hough_lines/'
OUTPUT_DIR_CIRCLES = 'images/output/hough_circles/'


def ensure_directories():
    os.makedirs(OUTPUT_DIR_LINES, exist_ok=True)
    os.makedirs(OUTPUT_DIR_CIRCLES, exist_ok=True)


def detect_hough_lines(img, gray):
    """Detect and draw Hough lines."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=30, maxLineGap=10)

    output = img.copy()
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(os.path.join(OUTPUT_DIR_LINES, 'lines_detected.jpg'), output)
    return output


def detect_hough_circles(img, gray):
    """Detect and draw Hough circles."""
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=450,
        param2=40,
        minRadius=20,
        maxRadius=80
    )

    output = img.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, r in circles[0, :]:
            cv2.circle(output, (x, y), r, (255, 0, 255), 2)  # Circle
            cv2.circle(output, (x, y), 2, (0, 255, 255), 3)  # Center

    cv2.imwrite(os.path.join(OUTPUT_DIR_CIRCLES, 'circles_detected.jpg'), output)
    return output


def show_image(title, img):
    """Optional display with matplotlib."""
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()


def main():
    ensure_directories()

    img = cv2.imread(INPUT_PATH)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lines_img = detect_hough_lines(img, gray)
    circles_img = detect_hough_circles(img, gray)

    # Optional display
    show_image("Hough Lines", lines_img)
    show_image("Hough Circles", circles_img)


if __name__ == "__main__":
    main()
