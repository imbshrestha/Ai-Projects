import cv2
import os

def main():
    # Path to the brain image file in the Downloads folder
    image_path = "/Users/bimalshrestha/Desktop/opencv/brain_image.jpg"

    # Read the image
    brain_image = cv2.imread(image_path)

    # Display the image
    cv2.imshow("Brain Image", brain_image)

    # Specify the directory on your desktop to save the copy
    desktop_directory = "/Users/bimalshrestha/Desktop/opencv"

    # Save a copy of the image to the desktop
    copy_path = os.path.join(desktop_directory, "brain_image_copy.jpg")
    cv2.imwrite(copy_path, brain_image)

    # Print a message indicating the path where the copy is saved
    print(f"Copy of the image saved to: {copy_path}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()

