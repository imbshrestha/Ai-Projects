import cv2
import numpy as np
import pytesseract


# Function to preprocess the plate region before character recognition
def preprocess_plate(plate_region):
    # Apply additional image processing steps for optimal character recognition
    # Example: Apply Gaussian blur
    plate_blurred = cv2.GaussianBlur(plate_region, (5, 5), 0)

    return plate_blurred


# Function to detect license plates in an image using a cascade classifier
def detect_license_plate(image_path, cascade_path):
    # Load the cascade classifier
    cascade_classifier = cv2.CascadeClassifier(cascade_path)

    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect license plates in the image
    plates = cascade_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))

    # Process each detected license plate
    for (x, y, w, h) in plates:
        # Extract the license plate region
        plate_region = gray[y:y + h, x:x + w]

        # Preprocess the plate region
        plate_preprocessed = preprocess_plate(plate_region)

        # Resize the plate region to a fixed size
        plate_resized = cv2.resize(plate_region, (300, 100))

        # Apply thresholding to the plate region
        _, plate_thresholded = cv2.threshold(plate_resized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Rotate and scale the plate region for horizontal alignment
        # Implement rotation and scaling here if needed

        # Perform character recognition using Tesseract OCR
        plate_text = pytesseract.image_to_string(plate_thresholded, config='--psm 8 --oem 3')

        # Draw a red boundary box around the detected plate
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Display the plate text
        cv2.putText(image, plate_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the image with the red boundary boxes and recognized plate text
    cv2.imshow('License Plate Detection and Recognition', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return plate_text


# Path to the cascade classifier for license plate detection
cascade_path = 'haarcascade_russian_plate_number.xml'

# Image paths
image_paths = [
    '/Users/bimalshrestha/Desktop/opencv/russian_lplate_image1.jpeg',
    '/Users/bimalshrestha/Desktop/opencv/russian_lplate_image2.jpeg',
    '/Users/bimalshrestha/Desktop/opencv/non_russian_lplate_image3.jpeg'
]

# Ground truth plate numbers for each image
ground_truth = {
    'russian_lplate_image1.jpeg': 'ABC123',
    'russian_lplate_image2.jpeg': 'DEF456',
    'non_russian_lplate_image3.jpeg': 'GHI789'
}

# Variables to store total plates and correct plates
total_plates = 0
correct_plates = 0

# Detect and recognize license plates in each image
for path in image_paths:
    # Extract the filename from the path
    filename = path.split('/')[-1]

    # Detect license plate
    detected_plate = detect_license_plate(path, cascade_path)

    # Increment total plates count
    total_plates += 1

    # Check if the detected plate matches the ground truth
    if ground_truth.get(filename) == detected_plate:
        correct_plates += 1

# Calculate and print accuracy
accuracy = (correct_plates / total_plates) * 100
print(f'Accuracy: {accuracy:.2f}%')
