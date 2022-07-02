import numpy as np
from PIL import Image

my_image = "lenna.png"

image = Image.open(my_image)

import matplotlib.pyplot as plt

plt.imshow(image)

# image.show(title="Lena")
# image.format:PNG