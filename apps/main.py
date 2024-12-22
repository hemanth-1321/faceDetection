import cv2
import pickle
import numpy as np

# Load the trained recognizer and label encoder
recognizer = cv2.face.LBPHFaceRecognizer_create()  # This should work now
recognizer.read('C:/Desktop/faceDetection/models/Attendance_model.yml')

with open('C:/Desktop/faceDetection/models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Initialize webcam
camera = cv2.VideoCapture(0)  # Use 0 for the default camera

# Load OpenCV's pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # For each detected face, predict the label
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face)  # Predict the label
        
        # Get the student name from the label
        student_name = label_encoder.inverse_transform([label])[0]
        
        # Display the name and a rectangle around the face
        cv2.putText(frame, f"Name: {student_name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show the frame with the face detected
    cv2.imshow('Face Recognition', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
camera.release()
cv2.destroyAllWindows()
