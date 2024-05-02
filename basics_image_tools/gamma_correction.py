import numpy as np
import cv2


def gammaCorrection(src, gamma):

    table = [((i / 255) ** gamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)


car = cv2.imread('ImageTrabalho1/car.png')
crowd = cv2.imread('ImageTrabalho1/crowd.png')
university = cv2.imread('ImageTrabalho1/university.png')


# Car Image
car_y_greater = gammaCorrection(car, 1.2)
car_y_less = gammaCorrection(car, 0.78)
cv2.imshow('Original image', np.uint8(car))
cv2.imshow('Gamma_corrected_car_Y = 1.2', np.uint8(car_y_greater))
cv2.imshow('Gamma_corrected_car_Y = 0.78', np.uint8(car_y_less))
cv2.imwrite('Questao_2_Image/Gamma_corrected_car__Y = 1.2.png', car_y_greater)
cv2.imwrite('Questao_2_Image/Gamma_corrected_car__Y = 0.78.png', car_y_less)


# Crowd Image
crowd_y_greater = gammaCorrection(crowd, 2)
crowd_y_less = gammaCorrection(crowd, 0.5)
cv2.imshow('Original image', np.uint8(crowd))
cv2.imshow('Gamma_corrected_crowd_Y = 2.png',
           np.uint8(crowd_y_greater))
cv2.imshow('Gamma_corrected_crowd_Y = 0.5.png',
           np.uint8(crowd_y_less))
cv2.imwrite('Questao_2_Image/Gamma_corrected_crowd__Y = 2.png',
            crowd_y_greater)
cv2.imwrite('Questao_2_Image/Gamma_corrected_crowd__Y = 0.5.png',
            crowd_y_less)

# University Image
university_y_greater = gammaCorrection(university, 3)
university_y_less = gammaCorrection(university, 0.5)
# cv2.imshow('Original image', np.uint8(university))
cv2.imshow('Gamma_corrected_university_Y = 3.png',
           np.uint8(university_y_greater))

cv2.imshow('Gamma_corrected_university_Y = 0.5.png',
           np.uint8(university_y_less))
cv2.imwrite('Questao_2_Image/Gamma_corrected_university__Y = 3.png',
            university_y_greater)
cv2.imwrite('Questao_2_Image/Gamma_corrected_university__Y = 0.5.png',
            university_y_less)

cv2.waitKey(0)
cv2.destroyAllWindows()
