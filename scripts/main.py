import cv2
import os
import face_recognition
import pickle
import datetime
import pandas as pd
import numpy as np

# --- Configuration ---
DATASET_DIR = "dataset"
ENCODING_DIR = "encodings"
ENCODING_FILE = os.path.join(ENCODING_DIR, "face_encodings.pkl")
ATTENDANCE_DIR = "attendance"
ATTENDANCE_FILE = os.path.join(ATTENDANCE_DIR, "attendance.csv")
FACE_ENCODING_MODEL = "hog" # or "cnn" (requires more computation)
FACE_RECOGNITION_TOLERANCE = 0.6 # Adjust for sensitivity

# --- Ensure directories exist ---
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(ENCODING_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

def capture_images():
    """Captures images from the webcam to create a dataset."""
    name = input("Enter Name: ")
    person_path = os.path.join(DATASET_DIR, name)
    os.makedirs(person_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    print(f"Press 's' to capture image for {name}, 'q' to finish capturing.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Capture Images", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            image_path = os.path.join(person_path, f"{count}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Image saved: {image_path}")
            count += 1
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Image capturing complete.")

def encode_faces():
    """Encodes faces from the dataset and saves the encodings."""
    known_encodings = []
    known_names = []

    print("🔄 Encoding registered faces...")

    for student_name in os.listdir(DATASET_DIR):
        student_path = os.path.join(DATASET_DIR, student_name)
        if os.path.isdir(student_path):
            for image_file in os.listdir(student_path):
                image_path = os.path.join(student_path, image_file)
                try:
                    image = cv2.imread(image_path)
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_image, model=FACE_ENCODING_MODEL)
                    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

                    if face_encodings:
                        known_encodings.append(face_encodings[0])
                        known_names.append(student_name)
                        print(f"✅ Encoded {student_name} from {image_file}")
                    else:
                        print(f"⚠️ No face detected in {image_path}")
                except Exception as e:
                    print(f"❌ Error processing {image_path}: {e}")

    data = {"encodings": known_encodings, "names": known_names}
    with open(ENCODING_FILE, "wb") as f:
        pickle.dump(data, f)

    print("🎉 Face encoding completed successfully!")

def recognize_faces_and_mark_attendance():
    """Recognizes faces from live video and marks attendance."""
    if not os.path.exists(ENCODING_FILE):
        print("❌ Error: Encoded face data not found! Run 'encode_faces' first.")
        return

    try:
        with open(ENCODING_FILE, "rb") as f:
            data = pickle.load(f)
        known_face_encodings = data["encodings"]
        known_face_names = data["names"]
    except Exception as e:
        print(f"❌ Error loading encodings: {e}")
        return

    print("📷 Starting camera for face recognition...")
    video_capture = cv2.VideoCapture(0)

    if not os.path.exists(ATTENDANCE_FILE):
        pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(ATTENDANCE_FILE, index=False)

    recognized_names = set()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model=FACE_ENCODING_MODEL)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=FACE_RECOGNITION_TOLERANCE)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances) if face_distances.size > 0 else None

            if best_match_index is not None and matches[best_match_index]:
                name = known_face_names[best_match_index]

            top, right, bottom, left = face_location
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            if name != "Unknown" and name not in recognized_names:
                recognized_names.add(name)
                now = datetime.datetime.now()
                new_entry = pd.DataFrame([[name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")]],
                                         columns=["Name", "Date", "Time"])
                try:
                    df = pd.read_csv(ATTENDANCE_FILE)
                    df = pd.concat([df, new_entry], ignore_index=True)
                    df.to_csv(ATTENDANCE_FILE, index=False)
                    print(f"✅ Attendance marked for {name} at {now.strftime('%H:%M:%S')}")
                except pd.errors.ParserError:
                    print("⚠ Corrupted CSV detected. Resetting attendance file.")
                    df = pd.DataFrame(columns=["Name", "Date", "Time"])
                    df.to_csv(ATTENDANCE_FILE, index=False)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    print("👋 Face recognition and attendance marking stopped.")

if __name__ == "__main__":
    while True:
        print("\nFace Recognition Attendance System Menu:")
        print("1. Capture Images for Dataset")
        print("2. Encode Faces from Dataset")
        print("3. Recognize Faces and Mark Attendance")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            capture_images()
        elif choice == '2':
            encode_faces()
        elif choice == '3':
            recognize_faces_and_mark_attendance()
        elif choice == '4':
            print("Exiting the system. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")