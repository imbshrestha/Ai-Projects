import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = '/Users/bimalshrestha/Desktop/opencv/bank_notes.jpg'
image = cv2.imread(image_path)

# Check if the image is loaded successfully
if image is None:
    print(f"Error: Unable to load image from {image_path}")
    exit()

# Display the original image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.show()

# Examine pixel matrix
rows, cols, channels = image.shape
print(f"Image dimensions: {rows} x {cols}")
print(f"Pixel value at (0, 0): {image[0, 0]}")

# Perform translation (move the image by a certain distance)
translation_matrix = np.float32([[1, 0, 50], [0, 1, 20]])
translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))

# Display the translated image
plt.imshow(cv2.cvtColor(translated_image, cv2.COLOR_BGR2RGB))
plt.title('Translated Image')
plt.show()
