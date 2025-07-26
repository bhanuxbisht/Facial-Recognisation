import cv2
from deepface import DeepFace

# Load the face cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale to RGB for DeepFace analysis
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Crop the face region from the frame
        face_roi = rgb_frame[y:y + h, x:x + w]

        # Analyze the emotion of the face using DeepFace
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Extract the dominant emotion from the result
        emotion = result[0]['dominant_emotion']

        # Draw a rectangle around the face and display the emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the frame with the detected face and emotion
    cv2.imshow('Real-time Emotion Detection', frame)

    # Exit the loop if the user presses the 'e' key
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()