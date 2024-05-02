import numpy as np
import cv2
import time


def TAM2(image_original: np.ndarray, n: int):
    rows, columns, channels = image_original.shape
    nova_rows = rows * n
    nova_columns = columns * n

    new_image = np.full((nova_rows, nova_columns, channels), fill_value=-1)

    new_image[::n, ::n, :] = image_original

    for i in range(1, n):
        new_image[i::n, :, :] = new_image[i-1::n, :, :]

    for j in range(1, n):
        new_image[:, j::n, :] = new_image[:, j-1::n, :]

    return new_image


original_image = cv2.imread('basics_image_tools/images/fruit1.jpg')
new_image = TAM2(original_image, 8)
print(new_image.shape)
cv2.imshow('Pixelate Image', np.uint8(new_image))

cv2.imshow("Original Image", np.uint8(original_image))
cv2.waitKey(0)
time.sleep(2)
cv2.destroyAllWindows()
