"""
Summary:
This Python program applies mean, median, and Gaussian filters to an image containing impulse noise.
It loads the image, applies the filters with different kernel sizes, and plots the original image and
filtered images side by side for comparison. The program provides flexibility to choose different kernel
sizes for mean and median filters and also allows applying Gaussian filtering with different sigma values.
The purpose of the program is to visually compare the effects of different filters on the image and
determine the most suitable filtering technique for noise removal.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "/Users/bimalshrestha/Desktop/opencv/impulse_noise.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply filters with different kernel sizes
kernel_sizes = [3, 5, 7]
sigmas = [1, 3]  # Sigma values for Gaussian filter

plt.figure(figsize=(12, 9))

for i, kernel_size in enumerate(kernel_sizes):
    for j, filter_type in enumerate(["Mean", "Median", "Gaussian (sigma=1)", "Gaussian (sigma=3)"]):
        plt.subplot(3, 4, i*4 + j + 1)

        # Apply mean filter
        if filter_type == "Mean":
            filtered_image = cv2.blur(image, (kernel_size, kernel_size))
        # Apply median filter
        elif filter_type == "Median":
            filtered_image = cv2.medianBlur(image, kernel_size)
        # Apply Gaussian filter
        else:
            sigma = sigmas[j-2] if filter_type == "Gaussian (sigma=3)" else sigmas[0]
            filtered_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

        plt.imshow(filtered_image, cmap='gray')
        plt.title(filter_type + " ({}x{})".format(kernel_size, kernel_size))
        plt.axis('off')

plt.tight_layout()
plt.show()
