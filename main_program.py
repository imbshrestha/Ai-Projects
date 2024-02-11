# main_program.py
import dlib
import cv2
import numpy as np
from image_utils import load_image, resize_image, blend_images

def get_face_landmarks(image_path):
    """Detect facial landmarks using dlib."""
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this file from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

    image = load_image(image_path)
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    if not faces:
        print("No faces detected.")
        return None

    shape = predictor(gray, faces[0])
    landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(shape.num_parts)], dtype=np.int32)

    return landmarks

def align_faces(image1_path, image2_path):
    """Align facial features of two images."""
    landmarks1 = get_face_landmarks(image1_path)
    landmarks2 = get_face_landmarks(image2_path)

    if landmarks1 is None or landmarks2 is None:
        return None

    # Load images
    image1 = load_image(image1_path)
    image2 = load_image(image2_path)

    # Resize images to have the same dimensions (optional)
    image1 = resize_image(image1, image2.shape[1], image2.shape[0])

    # Calculate the affine transformation matrix
    transformation_matrix = cv2.estimateAffine2D(landmarks1, landmarks2)[0]

    # Apply the transformation to image1
    aligned_image1 = cv2.warpAffine(image1, transformation_matrix, (image2.shape[1], image2.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return aligned_image1

def main():
    # Specify the paths of the images
    image1_path = "/Users/bimalshrestha/Desktop/opencv/bimal.jpg"
    image2_path = "/Users/bimalshrestha/Desktop/opencv/monica.jpg"

    # Align facial features
    aligned_image1 = align_faces(image1_path, image2_path)

    if aligned_image1 is not None:
        # Set the blending factor (alpha value)
        blending_alpha = 0.5

        # Blend images
        blended_image = blend_images(aligned_image1, load_image(image2_path), blending_alpha)

        # Display the images
        cv2.imshow("Aligned Image 1", aligned_image1)
        cv2.imshow("Image 2", load_image(image2_path))
        cv2.imshow("Blended Image", blended_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
