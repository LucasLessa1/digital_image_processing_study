import cv2
import numpy as np

# Load the grayscale image
gray_image = cv2.imread('Trabalho_2/imagens_p2/img_cells.jpg',
                        cv2.IMREAD_GRAYSCALE)

# 3.1: Apply morphological closing with an elliptical kernel
ellipse_kernel_8x8 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
closed_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE,
                                ellipse_kernel_8x8)

# Compute the difference to remove background
background_removed_image = gray_image - closed_image

# Apply Otsu thresholding
ellipse_kernel_5x5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_, thresholded_image = cv2.threshold(
    gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# 3.3: Fill holes in the binary image
inverted_thresholded_image = 255 - thresholded_image

# Find external contours and fill them to remove holes
contours, _ = cv2.findContours(
    inverted_thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
)
contour_mask = np.zeros_like(inverted_thresholded_image)
cv2.fillPoly(contour_mask, contours, 255)
holes_filled_image = 255 - contour_mask

# Address unfilled edge cells
num_labels, labels = cv2.connectedComponents(holes_filled_image)
edge_cells_mask = np.array(labels, dtype=np.uint8)
edge_cells_mask[labels == 1] = 255

# 3.4: Apply dilation and erosion
ellipse_kernel_2x2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
dilated_and_eroded_image = cv2.dilate(edge_cells_mask, ellipse_kernel_2x2,
                                      iterations=2)
final_morphed_image = cv2.erode(dilated_and_eroded_image, ellipse_kernel_2x2,
                                iterations=1)

# Apply distance transformation for the watershed algorithm
distance_transform = cv2.distanceTransform(final_morphed_image, cv2.DIST_L2,
                                           cv2.DIST_MASK_PRECISE)

# 3.5: Implement watershed segmentation
watershed_threshold = 0.1 * distance_transform.max()
_, sure_foreground = cv2.threshold(distance_transform,
                                   watershed_threshold, 255, 0)

sure_foreground = np.uint8(sure_foreground)
sure_dilated = cv2.dilate(sure_foreground, ellipse_kernel_2x2,
                          iterations=3)
unknown_region = cv2.subtract(sure_foreground, sure_dilated)

# Determine markers for watershed
_, markers = cv2.connectedComponents(sure_foreground)
markers += 1  # Increment to avoid conflicts
markers[unknown_region == 255] = 0  # Mark unknown regions

# Perform watershed
watershed_image = cv2.watershed(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR),
                                markers)

# Assign colors to different regions
region_colors = [
    (np.random.randint(0, 256), np.random.randint(0, 256),
     np.random.randint(0, 256)) for _ in range(1, np.max(markers) + 1)
]

# Create the final colorized segmentation image
colored_watershed_image = np.zeros(
    (gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8
)

for label in np.unique(watershed_image):
    if label == -1:
        continue  # Skip boundary regions
    mask = watershed_image == label
    colored_watershed_image[mask] = region_colors[label - 1]

# Display the results
cv2.imshow("Original Image",
           gray_image)
cv2.imshow("Background Removed",
           background_removed_image)
cv2.imshow("Thresholded",
           thresholded_image)
cv2.imshow("Holes Filled",
           holes_filled_image)
cv2.imshow("Distance Transform",
           distance_transform)
cv2.imshow("Watershed Result",
           colored_watershed_image)

cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()  # Close all OpenCV windows