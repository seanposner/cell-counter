import cv2
import numpy as np
from PIL import Image
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def select_image_file():
    """
    Opens a file dialog to select an image file and returns its path.
    """
    root = Tk()
    root.withdraw()  # Hide the main window

    file_path = askopenfilename(
        title="Select Image File",
        filetypes=[("Image Files", "*.tiff;*.tif;*.jpg;*.png"), ("All Files", "*.*")]
    )
    root.destroy()
    return file_path

def compute_average_cell(image, contours, min_area=10, cell_size=(50, 50)):
    """
    Computes and returns an "average cell" image from the given list of contours.
      - image: The original BGR image (np.array).
      - contours: List of detected contours.
      - min_area: Minimum contour area to consider a valid cell.
      - cell_size: (width, height) to which each cell is resized before averaging.
    Returns:
      - average_cell (np.array) of shape cell_size in BGR, or None if no valid cells.
    """
    valid_cells = []
    w, h = cell_size

    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            x, y, cw, ch = cv2.boundingRect(cnt)
            # Extract the cell region from the image
            cell_crop = image[y:y+ch, x:x+cw]
            # Resize to the standard cell size
            resized_cell = cv2.resize(cell_crop, (w, h), interpolation=cv2.INTER_AREA)
            valid_cells.append(resized_cell.astype(np.float32))

    if not valid_cells:
        return None

    # Accumulate all valid cells
    acc = np.zeros((h, w, 3), dtype=np.float32)
    for cell in valid_cells:
        acc += cell

    # Compute the average
    acc /= len(valid_cells)
    # Convert back to uint8
    average_cell = acc.astype(np.uint8)
    return average_cell

def count_and_mark_cells(image_bgr, target_color_bgr, tolerance, min_area=10):
    """
    Counts and marks cells based on color tolerance in HSV space.
    Returns:
      - marked_image: The BGR image with contours drawn and labeled.
      - contours: The list of all contours found (before area filtering).
      - valid_contours: The list of contours above the min_area threshold.
    """
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Convert the target BGR color to HSV
    target_color_hsv = cv2.cvtColor(
        np.uint8([[target_color_bgr]]), cv2.COLOR_BGR2HSV
    )[0][0]

    # Define a color range around the target color with the given tolerance
    lower_range = np.clip(target_color_hsv - tolerance, 0, 255)
    upper_range = np.clip(target_color_hsv + tolerance, 0, 255)

    # Create a mask to isolate the targeted color areas
    mask = cv2.inRange(hsv, lower_range, upper_range)

    # Find contours on the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours and number them on the original image
    marked_image = image_bgr.copy()
    valid_contours = []
    cell_index = 0

    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cell_index += 1
            valid_contours.append(contour)
            cv2.drawContours(marked_image, [contour], -1, (0, 255, 0), 2)
            cv2.putText(
                marked_image,
                str(cell_index),
                tuple(contour[0][0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )

    return marked_image, contours, valid_contours

def is_close_to_cell(point, cells, threshold=10):
    """
    Checks if a point is close to any cell center in 'cells' within a given threshold.
    Returns a tuple (bool, cell_point) indicating whether it's close, and to which cell.
    """
    px, py = point
    for cell in cells:
        cx, cy = cell
        dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        if dist < threshold:
            return True, (cx, cy)
    return False, None

def main():
    # Select the image file
    image_path = select_image_file()
    if not image_path:
        print("No file selected.")
        return

    # Load image via PIL, convert to OpenCV BGR
    pil_image = Image.open(image_path)
    image_np = np.array(pil_image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Define target color in BGR and tolerance in HSV
    target_color_bgr = [87, 29, 35]
    tolerance = np.array([15, 40, 35])  # in HSV

    # Automatic cell detection
    marked_image, all_contours, valid_contours = count_and_mark_cells(
        image_bgr, 
        target_color_bgr, 
        tolerance,
        min_area=10
    )

    # Compute and show average cell (optional)
    average_cell = compute_average_cell(image_bgr, valid_contours, min_area=10, cell_size=(50, 50))
    if average_cell is not None:
        cv2.namedWindow("Average Cell")
        cv2.imshow("Average Cell", average_cell)
    else:
        print("No valid cells found for averaging.")

    # Use a dictionary to store shared state so we can avoid using global vars
    state = {
        "cell_count": len(valid_contours),   # initial count is from automatic detection
        "manual_cells": [],                  # list of (x, y) for manually added cells
        "marked_image": marked_image.copy(), # the image we’ll show and modify
        "original_image": image_bgr.copy(),  # unmodified original for re-drawing
        "all_contours": all_contours,        # all raw contours
        "valid_contours": valid_contours     # automatically detected valid contours
    }

    # Create a separate window for the counter
    cv2.namedWindow("Counter")

    def update_counter_window():
        """
        Updates the "Counter" window with the current cell count from the state dictionary.
        """
        counter_image = np.zeros((100, 250, 3), dtype=np.uint8)
        text = f"Cell Count: {state['cell_count']}"
        cv2.putText(
            counter_image, 
            text, 
            (10, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        cv2.imshow("Counter", counter_image)

    update_counter_window()

    def redraw_marked_image():
        """
        Redraws the automatic detection and manual additions on top of the original image.
        """
        # First re-run the automatic detection to get the contours drawn
        auto_marked, _, _ = count_and_mark_cells(
            state["original_image"],
            target_color_bgr,
            tolerance,
            min_area=10
        )
        # Then draw manual circles
        for i, (mx, my) in enumerate(state["manual_cells"], start=len(state["valid_contours"]) + 1):
            cv2.circle(auto_marked, (mx, my), 10, (0, 0, 255), 2)
            cv2.putText(
                auto_marked,
                str(i),
                (mx, my),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )
        state["marked_image"] = auto_marked

    def mouse_callback(event, x, y, flags, param):
        """
        Mouse callback for counting cells manually.
        Left-click  -> add a new cell
        Right-click -> remove cell if close to an existing manual cell
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add a cell
            state["cell_count"] += 1
            state["manual_cells"].append((x, y))
            redraw_marked_image()
            update_counter_window()

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove a cell if close
            close, cell_coord = is_close_to_cell((x, y), state["manual_cells"], threshold=10)
            if close:
                state["cell_count"] -= 1
                state["manual_cells"].remove(cell_coord)
                redraw_marked_image()
                update_counter_window()

    # Set up the main "Image" window and assign the mouse callback
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)

    # Main display loop
    while True:
        cv2.imshow("Image", state["marked_image"])
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to quit
            break

    # Cleanup
    cv2.destroyAllWindows()

    # Print final count
    print("Final cell count:", state["cell_count"])

if __name__ == "__main__":
    main()
