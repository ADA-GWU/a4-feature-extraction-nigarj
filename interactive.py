import numpy as np
import cv2
import os
from skimage import exposure
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour


OUTPUT_DIR = 'images/output/active_contour/'


# === Drawing UI ===
class ContourDrawer:
    def __init__(self, window_name, image):
        self.window_name = window_name
        self.original = image.copy()
        self.image = image.copy()
        self.drawing = False
        self.points = []

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.points = [(x, y)]
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.points.append((x, y))
            cv2.line(self.image, self.points[-2], self.points[-1], (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.points.append((x, y))
            cv2.line(self.image, self.points[-2], self.points[-1], (0, 255, 0), 2)
            cv2.line(self.image, self.points[-1], self.points[0], (0, 255, 0), 2)
    def draw_instructions(self, frame):
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # --- Add translucent black banner at the top ---
        banner_height = 60
        cv2.rectangle(overlay, (0, 0), (w, banner_height), (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # --- Write instructions in the banner ---
        instructions = [
            "Draw initial contour with left mouse drag",
            "Press SPACE to run active contour",
            "Press any key close window to exit"
        ]
        for i, text in enumerate(instructions):
            y = 20 + i * 18
            cv2.putText(frame, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame


    def run(self):
        print("[i] Draw initial contour. Press SPACE to process, R to reset, or any key to close window to exit.")
        while True:
            gui_frame = self.draw_instructions(self.image.copy())
            cv2.imshow(self.window_name, gui_frame)
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                return False
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                self.image = self.original.copy()
                self.points = []
            elif key == 32 and self.points:
                return True


# === Active Contour Logic ===
def preprocess_image(image):
    gray = rgb2gray(image)
    contrast = exposure.equalize_adapthist(gray)
    return gaussian(contrast, sigma=1)


def run_active_contour(image, initial_pts, max_iter=5000):
    processed = preprocess_image(image)
    init_array = np.fliplr(np.array(initial_pts))
    snake = active_contour(
        processed,
        init_array,
        alpha=0.01,
        beta=1.0,
        gamma=0.01,
        max_num_iter=max_iter,
        convergence=0.01,
        boundary_condition='periodic'
    )
    return np.fliplr(snake).astype(np.int32)


def animate_mask_transition(image, initial, final, steps=500):
    print("[i] Animating mask evolution...")

    initial = np.array(initial, dtype=np.float32)
    final = np.array(final, dtype=np.float32)

    for t in np.linspace(0, 1, steps):
        interp = (1 - t) * initial + t * final
        interp = interp.astype(np.int32)

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [interp], 255)

        color_mask = np.zeros_like(image)
        color_mask[:, :, 1] = mask  # Green channel

        blend = cv2.addWeighted(image, 1.0, color_mask, 0.5, 0)
        cv2.imshow("Contour Evolution", blend)

        if cv2.getWindowProperty("Contour Evolution", cv2.WND_PROP_VISIBLE) < 1:
            break
        cv2.waitKey(10)

    cv2.destroyWindow("Contour Evolution")


def save_final_contour(image, contour, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result_img = image.copy()
    cv2.polylines(result_img, [contour], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), result_img)


def main(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[!] Failed to load image at: {image_path}")
        return

    drawer = ContourDrawer("Interactive Contour Drawing", image)
    if not drawer.run():
        print("[!] Window closed. Exiting.")
        return

    if not drawer.points:
        print("[!] No points selected. Exiting.")
        return

    final_snake = run_active_contour(image, drawer.points)
    animate_mask_transition(image, drawer.points, final_snake)

    save_final_contour(image, final_snake, 'pear_contour_result.jpg')

    # Show final result
    result_img = image.copy()
    cv2.polylines(result_img, [final_snake], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.imshow("Final Contour Boundary", result_img)

    print("[i] Press any key or close the window to exit.")
    while True:
        if cv2.getWindowProperty("Final Contour Boundary", cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(1) != -1:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main("images/original/pear.jpeg")
