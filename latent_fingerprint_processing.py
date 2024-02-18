import cv2
import numpy as np

# Load the image
image_path = "/Users/bimalshrestha/Desktop/opencv/sample-latent-fingerprint.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply morphological operations
kernel = np.ones((5, 5), np.uint8)  # Define a 5x5 kernel
dilated_image = cv2.dilate(image, kernel, iterations=1)  # Dilate the image
eroded_image = cv2.erode(dilated_image, kernel, iterations=1)  # Erode the dilated image
opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)  # Perform opening operation
closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)  # Perform closing operation

# Display the processed images
cv2.imshow("Original Image", image)
cv2.imshow("Dilated Image", dilated_image)
cv2.imshow("Eroded Image", eroded_image)
cv2.imshow("Opened Image", opened_image)
cv2.imshow("Closed Image", closed_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
