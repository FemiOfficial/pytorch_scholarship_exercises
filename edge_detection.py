import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
import numpy as np



# Read in the image
image = mpimg.imread('data/curved_lane.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# Convert to grayscale for filtering

# 3x3 array for edge detection
sobel_y = np.array([[ -1, -2, -1],
                   [ 0, 0, 0],
                   [ 1, 2, 1]])

## TODO: Create and apply a Sobel x operator


# Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)
filtered_image = cv2.filter2D(gray, -1, sobel_y)

plt.imshow(filtered_image, cmap='gray')
plt.show()