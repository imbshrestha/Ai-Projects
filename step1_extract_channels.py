import cv2
import numpy as np

def extract_channels(image_path):
    # Load the colored image
    image = cv2.imread(image_path)

    # Extract individual color channels
    blue_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    red_channel = image[:, :, 2]

    return blue_channel, green_channel, red_channel

if __name__ == "__main__":
    image_path = '/Users/bimalshrestha/Desktop/opencv/puppy.jpg'

    blue_channel, green_channel, red_channel = extract_channels(image_path)

    # Display or save the individual channel images if needed
    # For demonstration, let's visualize the green channel
    cv2.imshow('Green Channel', green_channel)
    cv2.imshow('blue_channel', blue_channel)
    cv2.imshow('red_channel', red_channel)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

