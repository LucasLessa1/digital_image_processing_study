import numpy as np
import cv2
import matplotlib.pyplot as plt

car = cv2.imread('basics_image_tools/images/car.png', cv2.IMREAD_GRAYSCALE)
plt.hist(car.ravel(), 256, [0, 256])
plt.show()

# Car Image
car_eq = cv2.equalizeHist(car)
plt.hist(car_eq.ravel(), 256, [0, 256])
plt.show()


def plot_cdf(image):
    # Calculate the histogram of the image
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    # Calculate the cumulative sum of the histogram
    cdf = hist.cumsum()

    # Normalize the cumulative sum to [0, 1]
    cdf_normalized = cdf / cdf.max()

    # Plot the CDF
    plt.plot(cdf_normalized, color='b')
    plt.xlim([0, 256])
    plt.ylim([0, 1.25])
    plt.title('Imagem não equalizada')
    plt.xlabel('Pixel')
    plt.ylabel('CDF')
    plt.show()


# Plot the CDF of the image
plot_cdf(car)
plot_cdf(car_eq)