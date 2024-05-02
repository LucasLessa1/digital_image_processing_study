import cv2
import numpy as np

# Load the grayscale image
gray_image = cv2.imread(
             'Trabalho_2/imagens_p2/morf_test.png',
             cv2.IMREAD_GRAYSCALE
)

# Apply a median blur to reduce noise
blurred_image = cv2.medianBlur(gray_image, 5)

# Apply adaptive thresholding to create a binary image
binary_image = cv2.adaptiveThreshold(
    blurred_image, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11, 2
)

# Define a structuring element for morphological operations
ellipse_kernel_4x4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

# Perform dilation followed by erosion (closing)
dilated_image = cv2.dilate(binary_image, ellipse_kernel_4x4, iterations=1)
closed_image = cv2.erode(dilated_image, ellipse_kernel_4x4, iterations=1)

# Calculate the difference between the original and closed images
image_difference = closed_image - gray_image

# Apply bilateral filtering to remove noise while preserving edges
filtered_image = cv2.bilateralFilter(gray_image, 10, 80, 80)

# Apply adaptive thresholding to the filtered image
filtered_binary_image = cv2.adaptiveThreshold(
    filtered_image, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11, 2
)

# Invert the binary image
inverted_binary_image = 255 - filtered_binary_image

# Modify the structuring element for opening operation
ellipse_kernel_2by2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

# Perform opening (erosion followed by dilation) to remove noise
opened_image = cv2.morphologyEx(
    inverted_binary_image,
    cv2.MORPH_OPEN,
    ellipse_kernel_2by2,
    iterations=1
)

# Perform additional erosion and closing operations
eroded_image = cv2.erode(opened_image, ellipse_kernel_2by2, iterations=1)
final_image = cv2.morphologyEx(
              eroded_image, cv2.MORPH_CLOSE,
              ellipse_kernel_2by2, iterations=1
)

# Display the final image
cv2.imshow("Final Image", np.uint8(final_image))

# Save the processed images to files
cv2.imwrite("Trabalho_2/imagem_quest_2/gray_image.png",
            gray_image)
cv2.imwrite("Trabalho_2/imagem_quest_2/blurred_image.png",
            blurred_image)
cv2.imwrite("Trabalho_2/imagem_quest_2/binary_image.png",
            binary_image)
cv2.imwrite("Trabalho_2/imagem_quest_2/closed_image.png",
            closed_image)
cv2.imwrite("Trabalho_2/imagem_quest_2/image_difference.png",
            image_difference)
cv2.imwrite("Trabalho_2/imagem_quest_2/filtered_image.png",
            filtered_image)
cv2.imwrite("Trabalho_2/imagem_quest_2/filtered_binary_image.png",
            filtered_binary_image)
cv2.imwrite("Trabalho_2/imagem_quest_2/inverted_binary_image.png",
            inverted_binary_image)
cv2.imwrite("Trabalho_2/imagem_quest_2/opened_image.png",
            opened_image)
cv2.imwrite("Trabalho_2/imagem_quest_2/eroded_image.png",
            eroded_image)
cv2.imwrite("Trabalho_2/imagem_quest_2/final_image.png",
            final_image)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
