import threading
import cv2
import face_recognition
from datetime import datetime

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False
reference_image = "Elon Musk.jpg"
reference_img = face_recognition.load_image_file("Elon Musk.jpg")
reference_encoding = face_recognition.face_encodings(reference_img)[0]


def check_face(frame):
    global face_match
    try:
        # Find all face locations in the frame
        face_locations = face_recognition.face_locations(frame)

        if face_locations:
            # Encode the first detected face in the frame
            frame_encoding = face_recognition.face_encodings(frame, face_locations)[0]

            # Compare face encodings
            matches = face_recognition.compare_faces([reference_encoding], frame_encoding, tolerance=0.6)

            # Update face_match based on the result
            face_match = any(matches)

            if face_match:
                # Print the time and reference photo name
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"Match found at {current_time} with reference photo: {reference_image}")
        else:
            face_match = False
    except Exception as e:
        print(f"Error in face recognition: {e}")
        face_match = False


while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            threading.Thread(target=check_face, args=(frame.copy(),)).start()

        counter += 1

        # Draw rectangle based on face_match
        color = (0, 255, 0) if face_match else (0, 0, 255)
        thickness = 2

        # Find all face locations in the frame
        face_locations = face_recognition.face_locations(frame)

        # Draw rectangles around the faces
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
            if face_match:
                # Add text with the name of the reference photo
                cv2.putText(frame, f"Match: {reference_image}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
