
#  Assignment  4
**Feature Detection and Shape Analysis**

## Overview

This assignment demonstrates practical applications of classical computer vision techniques, including edge and corner detection, Hough transforms for line and circle detection, interactive active contour models, and robust feature matching. Each component is implemented in Python using OpenCV and other scientific computing libraries.

The project is modularized into five scripts:

- `corner_detection.py`
- `edge_detection.py`
- `interactive.py`
- `line_and_circle_detection.py`
- `orb.py`

All input images are organized under `images/original/`, with results stored in structured subdirectories for clarity and reproducibility.

---

## Directory Structure

```
project_root/
│
├── corner_detection.py
├── edge_detection.py
├── interactive.py
├── line_and_circle_detection.py
├── orb.py
│
├── images/
│   ├── original/           # Original input images
│   ├── edges/              # Output from edge detection
│   ├── corners/            # Output from corner detection
│   ├── hough_lines/        # Line detection results
│   ├── hough_circles/      # Circle detection results
│   ├── warped/             # Warped images for feature matching
│   └── output/             # Final result images and match logs
```

---

## Requirements

Install dependencies via `pip`:

```bash
pip install opencv-python numpy matplotlib scikit-image
```

---

##  Modules

### 1. `edge_detection.py`
Applies classical edge detection techniques (Canny, Sobel, Laplacian) to grayscale images.
- Outputs edge maps under `images/edges/`.

### 2. `corner_detection.py`
Implements Harris corner detectors.
- Detected corners are overlaid on images and saved in `images/corners/`.

### 3. `line_and_circle_detection.py`
Performs:
- **Line detection** using Hough Line Transform
- **Circle detection** using Hough Circle Transform
- Results are saved under `images/hough_lines/` and `images/hough_circles/`.

### 4. `interactive.py`
An interactive tool for manually initializing active contours (snakes):
- Users draw an initial contour on any image.
- The script applies active contour optimization and animates the mask evolution.
- Final contours are visualized with overlaid boundaries.

### 5. `orb.py`
Performs feature detection and matching:
- Uses ORB (Oriented FAST and Rotated BRIEF) to detect keypoints.
- Applies perspective warping to simulate transformations.
- Visualizes and logs match distances.
- Output images and logs are saved under `images/warped/` and `images/output/`.

---

##  Usage

Run each script individually:

```bash
python edge_detection.py
python corner_detection.py
python line_and_circle_detection.py
python interactive.py
python orb.py
```

Make sure your input images are stored in `images/original/`. Outputs will be automatically saved in their respective folders.

---

##  Notes

- This project focuses on classical (non-deep learning) image processing methods.
- Parameters (e.g., thresholds, radii) are adjustable in each script for fine-tuning.
- The active contour tool (`interactive.py`) requires user interaction to initialize the snake.

As usual, you can find the result analysis in the Project_Findings.pdf
