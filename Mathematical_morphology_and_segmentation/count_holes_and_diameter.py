import cv2
import numpy as np
from skimage.measure import label, regionprops


# Carregar a imagem
image = cv2.imread(
        'Mathematical_morphology_and_segmentation/images/pcb.jpg',
        cv2.IMREAD_GRAYSCALE
)

# Binarizar a imagem usando um valor de limiar
_, threshold_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Achar os contornos na imagem binária
contours, _ = cv2.findContours(
              image, cv2.RETR_EXTERNAL,
              cv2.CHAIN_APPROX_SIMPLE)

# Mascara toda preta
mask = np.zeros_like(image)

# Preenche os contornos com branco
cv2.fillPoly(mask, contours, 255)

diff = mask - threshold_image

# Fechamento
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

image_dilate = cv2.dilate(diff, kernel, iterations=1)
image_erode = cv2.erode(image_dilate, kernel, iterations=4)

image_dilate_2 = cv2.dilate(image_erode, kernel, iterations=1)
image_erode_2 = cv2.erode(image_dilate_2, kernel, iterations=6)

final_image = cv2.dilate(image_erode_2, kernel, iterations=8)

# label function names the connected regions
label_img = label(final_image, connectivity=2, background=0)


props = regionprops(label_img)

print(f'Quantidade de buracos: {len(props)}')

for i, prop in enumerate(props):

    diâmetro = prop.equivalent_diameter
    print("Buraco", i+1, "- Diâmetro:", diâmetro, "pixels")

cv2.imshow("Image", np.uint8(image))
cv2.imshow("Threshold Image", np.uint8(threshold_image))
cv2.imshow("Mask Image", np.uint8(mask))
cv2.imshow("Different =  mask - threshold_image", np.uint8(diff))
cv2.imshow("Close - dilate", np.uint8(image_dilate))
cv2.imshow("Close - erode", np.uint8(image_erode))
cv2.imshow("Close - dilate 2", np.uint8(image_dilate_2))
cv2.imshow("Close - erode 2", np.uint8(image_erode_2))
cv2.imshow("Final Image", np.uint8(final_image))


cv2.waitKey(0)  # Espera eu clicar em alguma tecla
cv2.destroyAllWindows()  # Apaga as janelas
