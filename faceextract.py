import cv2
import os

def detect_and_crop_faces(input_folder, output_folder, output_size=(128, 128)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Unable to load image {filename}.")
            continue
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            print(f"No face detected in {filename}.")
            continue
        
        for i, (x, y, w, h) in enumerate(faces):
            face = image[y:y+h, x:x+w]
            face_resized = cv2.resize(face, output_size)
            output_path = os.path.join(output_folder, f"cropped_{i}_{filename}")
            cv2.imwrite(output_path, face_resized)
            print(f"Saved cropped face to {output_path}")

# Example usage
input_folder = "/home/sid/ml-utils/input"  # Replace with the path to your folder containing images
output_folder = "/home/sid/ml-utils/foutput"  # Replace with the desired output folder path
detect_and_crop_faces(input_folder, output_folder)
