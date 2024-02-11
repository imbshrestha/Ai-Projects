import cv2
import numpy as np
from step1_extract_channels import extract_channels

def swap_red_green(image_path):
    # Extract channels
    blue_channel, green_channel, red_channel = extract_channels(image_path)

    # Swap red and green channels
    swapped_image = np.stack([blue_channel, red_channel, green_channel], axis=-1)

    return swapped_image

if __name__ == "__main__":
    image_path = '/Users/bimalshrestha/Desktop/opencv/puppy.jpg'

    swapped_image = swap_red_green(image_path)

    # Display or save the swapped image if needed
    cv2.imshow('Swapped Image (GRB)', swapped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
