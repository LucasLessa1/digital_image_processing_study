import numpy as np
import cv2
import time
import math


def TAMM(original_image):
    row, column, channel = original_image.shape
    new_image = np.full((row * 2, column * 2, channel), fill_value=0)
    # Take all rows and columns with a step equal 2
    new_image[0::2, 0::2, :] = original_image

    # Nesta faço com que o index do canal acompanha os índices iguais -1
    indices = np.argwhere(new_image == 0)
    for ind in indices:
        i, j, ch = ind
        # Somente linhas pares (0, 2, 4, 6, 8, ..., 256)
        if i % 2 == 0:

            before_e = new_image[i, j-1, ch]

            # The last column is equal to the before
            if j == len(new_image[:, 0, :])-1:
                next_e = before_e
            else:
                next_e = new_image[i, j+1, ch]

            new_image[i, j, ch] = math.ceil(np.mean([before_e, next_e]))

    for ind in indices:
        i, j, ch = ind
        if i % 2 != 0:
            before_e = new_image[i-1, j, ch]

            if i == len(new_image[:, 0, :])-1:
                next_e = before_e
            else:
                next_e = new_image[i+1, j, ch]

            new_image[i, j, ch] = math.ceil(np.mean([before_e, next_e]))

    return new_image


original_image = cv2.imread('basics_image_tools/images/fruit1.jpg')

new_image = TAMM(original_image)

cv2.imshow('Bigger Image with TAMM', np.uint8(new_image))
cv2.imshow("Original Image", np.uint8(original_image))
# cv2.imwrite(f"put_the_name_here.jpg", new_image)
cv2.waitKey(0)  # Espera eu clicar em alguma tecla
time.sleep(2)
cv2.destroyAllWindows()  # Apaga as janelas
