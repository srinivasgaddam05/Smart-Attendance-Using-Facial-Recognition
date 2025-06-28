import face_recognition
import pickle
import os
import cv2
import numpy as np

# Paths
DATASET_DIR = r"C:\Users\DELL\Downloads\Smart_Attendance_Using_Facial_Recognition\dataset"
ENCODING_FILE = r"C:\Users\DELL\Downloads\Smart_Attendance_Using_Facial_Recognition\encodings\face_encodings.pkl"

# Ensure encoding folder exists
os.makedirs(os.path.dirname(ENCODING_FILE), exist_ok=True)

known_encodings = []
known_names = []

print("🔄 Encoding registered faces...")

# Loop through dataset folder
for student_name in os.listdir(DATASET_DIR):
    student_path = os.path.join(DATASET_DIR, student_name)

    if os.path.isdir(student_path):
        for image_file in os.listdir(student_path):
            image_path = os.path.join(student_path, image_file)

            # Load and encode image
            image = cv2.imread(image_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_image)

            if face_encodings:
                known_encodings.append(face_encodings[0])
                known_names.append(student_name)
                print(f"✅ Encoded {student_name} from {image_file}")

# Save encodings
data = {"encodings": known_encodings, "names": known_names}
with open(ENCODING_FILE, "wb") as f:
    pickle.dump(data, f)

print("🎉 Face encoding completed successfully!")