import cv2
import os

# Create dataset directory if not exists
dataset_path = "dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

name = input("Enter Name: ")
person_path = os.path.join(dataset_path, name)

if not os.path.exists(person_path):
    os.makedirs(person_path)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture Images - Press 's' to save, 'q' to quit", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # Press 's' to save an image
        image_path = os.path.join(person_path, f"{count}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Image saved: {image_path}")
        count += 1

    elif key == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()