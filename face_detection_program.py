import PIL.Image
import PIL.ImageDraw
import face_recognition

# Load the images
individual_image_path = '/Users/bimalshrestha/Desktop/opencv/individual.jpeg'
group_image_path = '/Users/bimalshrestha/Desktop/opencv/groupimage.jpeg'
individual_image = face_recognition.load_image_file(individual_image_path)
group_image = face_recognition.load_image_file(group_image_path)

# Find face locations in both images
individual_face_locations = face_recognition.face_locations(individual_image)
group_face_locations = face_recognition.face_locations(group_image)

# Find face encodings in both images
individual_face_encodings = face_recognition.face_encodings(individual_image, individual_face_locations)
group_face_encodings = face_recognition.face_encodings(group_image, group_face_locations)

# Compare face encodings to check if the individual face is present in the group
for i, individual_face_encoding in enumerate(individual_face_encodings):
    for group_face_encoding in group_face_encodings:
        # Compare the face encodings
        match = face_recognition.compare_faces([group_face_encoding], individual_face_encoding)
        if match[0]:
            print(f"Individual face {i + 1} is present in the group.")
        else:
            print(f"Individual face {i + 1} is not present in the group.")
