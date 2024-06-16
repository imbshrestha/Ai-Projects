import PIL.Image
import PIL.ImageDraw
import face_recognition

# Load the image file
image_path = '/Users/bimalshrestha/Desktop/opencv/CrossWalk_(5465840138).jpeg'
image = face_recognition.load_image_file(image_path)

# Find all face locations in the image
face_locations = face_recognition.face_locations(image)

# Print the number of faces found
number_of_faces = len(face_locations)
print("Found {} face(s) in this picture.".format(number_of_faces))

# Load the image into a Python Image Library object so that you can draw on top of it and display it
pil_image = PIL.Image.fromarray(image)

# Iterate over each face found and draw a red box around it
for face_location in face_locations:
    top, right, bottom, left = face_location
    draw_handle = PIL.ImageDraw.Draw(pil_image)
    draw_handle.rectangle([left, top, right, bottom], outline="red")

# Display the image with the red boxes around the faces
pil_image.show()
