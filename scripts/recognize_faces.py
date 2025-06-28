import face_recognition
import pickle
import cv2
import numpy as np
import os
import datetime
import pandas as pd

# Paths
ENCODING_FILE = r"C:\Users\DELL\Downloads\Smart_Attendance_Using_Facial_Recognition\encodings\face_encodings.pkl"
ATTENDANCE_FILE = r"C:\Users\DELL\Downloads\Smart_Attendance_Using_Facial_Recognition\attendance\attendance.csv"

# Ensure encoding file exists
if not os.path.exists(ENCODING_FILE):
    print("❌ Error: Encoded face data not found! Run 'encode_faces.py' first.")
    exit()

# Load face encodings
with open(ENCODING_FILE, "rb") as f:
    data = pickle.load(f)

known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Initialize webcam
print("📷 Starting camera...")
video_capture = cv2.VideoCapture(0)

# Ensure attendance file exists with correct columns
os.makedirs(os.path.dirname(ATTENDANCE_FILE), exist_ok=True)

if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(ATTENDANCE_FILE, index=False)

# Track recognized names to prevent duplicate marking
recognized_names = set()

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare with known encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)  # Adjusted for accuracy
        name = "Unknown"

        # Find best match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances) if face_distances.size > 0 else None

        if best_match_index is not None and matches[best_match_index]:
            name = known_face_names[best_match_index]
        
        # Ensure unregistered faces are not falsely marked
        if name != "Unknown" and face_distances[best_match_index] > 0.5:
            name = "Unknown"

        # Draw a rectangle around the face
        top, right, bottom, left = face_location
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)  # Green for known, red for unknown
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Mark attendance if person is recognized and not already marked
        if name != "Unknown" and name not in recognized_names:
            recognized_names.add(name)
            now = datetime.datetime.now()
            new_entry = pd.DataFrame([[name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")]], 
                                     columns=["Name", "Date", "Time"])

            # Append to CSV safely
            try:
                df = pd.read_csv(ATTENDANCE_FILE)
                df = pd.concat([df, new_entry], ignore_index=True)
                df.to_csv(ATTENDANCE_FILE, index=False)
                print(f"✅ {name} marked present at {now.strftime('%H:%M:%S')}")
            except pd.errors.ParserError:
                print("⚠ Corrupted CSV detected. Resetting file...")
                df = pd.DataFrame(columns=["Name", "Date", "Time"])
                df.to_csv(ATTENDANCE_FILE, index=False)

    # Show the video frame
    cv2.imshow("Face Recognition", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
print("👋 Exiting Face Recognition System.")