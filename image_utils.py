# image_utils.py
import cv2
def load_image(image_path):
    """Load an image from the given file path."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image from {image_path}.")
    return image

def resize_image(image, target_width, target_height):
    """Resize the given image to the specified dimensions."""
    return cv2.resize(image, (target_width, target_height))

def blend_images(image1, image2, alpha=0.5):
    """Blend two images using the specified alpha value."""
    return cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)
