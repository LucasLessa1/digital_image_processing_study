import numpy as np
import cv2
import time
import math


def SUPERRES(image_1, image_2):

    rows, columns, channels = image_1.shape
    new_rows = rows * 2
    new_columns = columns * 2

    new_image = np.full((new_rows, new_columns, channels), fill_value=-1)
    new_image[::2, ::2, :] = image_1
    new_image[1::2, 1::2, :] = image_2

    # Nesta faço com que o index do canal acompanha os índices iguais -1
    indices = np.argwhere(new_image == -1)

    for ind in indices:
        i, j, ch = ind
        # Somente linhas pares (0, 2, 4, 6, 8, ..., 256)
        if i % 2 == 0:

            before_e = new_image[i, j-1, ch]
            upper = new_image[i - 1, j, ch]
            down = new_image[i+1, j, ch]

            if j == len(new_image[:, 0, :])-1:
                next_e = before_e
            else:
                next_e = new_image[i, j+1, ch]
            if i == 0 or j == 0 or i == new_rows or j == new_columns:
                new_image[i, j, ch] = math.ceil(np.mean([before_e, next_e]))

            else:
                new_image[i, j, ch] = math.ceil(np.mean([
                    before_e, next_e, upper, down]))

        else:

            before_r = new_image[i-1, j, ch]
            upper_c = new_image[i, j-1, ch]
            down = new_image[i, j+1, ch]

            if i == len(new_image[:, 0, :]) - 1:
                next_r = before_r
            else:
                next_r = new_image[i + 1, j, ch]

            if i == 0 or j == 0 or i == new_rows or j == new_columns:
                new_image[i, j, ch] = math.ceil(np.mean([before_r, next_r]))

            else:
                new_image[i, j, ch] = math.ceil(np.mean([
                    before_r, next_r, upper_c, down]))

    return new_image


image_1 = cv2.imread('ImageTrabalho1/fruit1.jpg')
image_2 = cv2.imread('ImageTrabalho1/fruit2.jpg')

new_image = SUPERRES(image_1, image_2)


cv2.imshow('Image - Fruits1 and Fruits2', np.uint8(new_image))
cv2.imshow("Fruits_1", np.uint8(image_1))
cv2.imshow("Fruits_2", np.uint8(image_2))

# cv2.imwrite(f"Quest_1_3_SUPERRES_fruits_1_e_2.jpg", np.uint8(new_image))
cv2.waitKey(0)  # Espera eu clicar em alguma tecla
time.sleep(2)
cv2.destroyAllWindows()  # Apaga as janelas
