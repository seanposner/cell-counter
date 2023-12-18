from PIL import Image
import numpy as np
import cv2

def count_and_mark_cells_with_tolerance(image_cv2, target_color_bgr, tolerance):
    # Convert the image to HSV color space for more effective color range matching
    hsv = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2HSV)

    # Convert the target BGR color to HSV
    target_color_hsv = cv2.cvtColor(np.uint8([[target_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]

    # Define a color range around the target color with the given tolerance
    lower_range = target_color_hsv - tolerance
    upper_range = target_color_hsv + tolerance

    # Ensure that HSV values are within valid ranges
    lower_range = np.clip(lower_range, 0, 255)
    upper_range = np.clip(upper_range, 0, 255)

    # Create a mask to isolate the targeted color areas
    mask = cv2.inRange(hsv, lower_range, upper_range)

    # Find contours on the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours and number them on the original image
    marked_image = image_cv2.copy()
    cell_count = 0
    for contour in contours:
        # Set a minimum size for the contours to be considered a cell
        if cv2.contourArea(contour) > 10:  # Minimum area threshold
            cell_count += 1
            cv2.drawContours(marked_image, [contour], -1, (0, 255, 0), 2)
            cv2.putText(marked_image, str(cell_count), tuple(contour[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return marked_image, cell_count

# Callback function for mouse events
def click_and_count(event, x, y, flags, param):
    global cell_count, marked_image_tolerance
    if event == cv2.EVENT_LBUTTONDOWN:
        # Increase the cell count and draw a circle to mark the manually selected cell
        cell_count += 1
        cv2.circle(marked_image_tolerance, (x, y), 5, (255, 0, 0), -1)
        cv2.putText(marked_image_tolerance, str(cell_count), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow("Image", marked_image_tolerance)

# Load the image
image_path = '/Users/seanposner/Downloads/CON BSA 2_RGB Trans.tiff'
image = Image.open(image_path)

# Convert the image to numpy array and then to OpenCV format
image_np = np.array(image)
image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

# # Display the image for visual reference
# image.show()

# Define the target color in BGR format and the tolerance
target_color_bgr = [87, 29, 35]
tolerance = np.array([50, 65, 65])  # Tolerance in HSV space

# Process the image with the new color range and tolerance
marked_image_tolerance, cell_count_tolerance = count_and_mark_cells_with_tolerance(image_cv2, target_color_bgr, tolerance)

# Convert the marked image back to PIL format for display
marked_image_tolerance_pil = Image.fromarray(cv2.cvtColor(marked_image_tolerance, cv2.COLOR_BGR2RGB))
marked_image_tolerance_pil.show()

# Display the image with OpenCV window
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", click_and_count)

# Keep the window open until 'ESC' key is pressed
while True:
    cv2.imshow("Image", marked_image_tolerance)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break
    
cv2.destroyAllWindows()

# Display the updated cell count
print(cell_count_tolerance)
