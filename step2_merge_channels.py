import cv2
import numpy as np
from step1_extract_channels import extract_channels

def merge_channels(blue_channel, green_channel, red_channel):
    # Merge channels to create a colored image
    colored_image = np.stack([blue_channel, green_channel, red_channel], axis=-1)

    return colored_image

if __name__ == "__main__":
    # Extract channels using the function from step1_extract_channels
    image_path = '/Users/bimalshrestha/Desktop/opencv/puppy.jpg'
    blue_channel, green_channel, red_channel = extract_channels(image_path)

    # Example: merging the channels back
    merged_image = merge_channels(blue_channel, green_channel, red_channel)

    # Display or save the merged image if needed
    cv2.imshow('Merged Image', merged_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
