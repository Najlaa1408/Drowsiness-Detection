from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
import pygame
import time

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('my_model.h5')

# Set the dimensions for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize pygame mixer for sound
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound('alert-sound.mp3')  # Replace with the path to your alarm sound file

# Define the region of interest (ROI) coordinates
roi_x, roi_y, roi_width, roi_height = 200, 200, 400, 400

def play_alarm():
    pygame.mixer.Sound.play(alarm_sound)

def generate_frames():
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    # Initialize variables for frames per second and drowsiness detection
    fps_counter = 0
    start_time = time.time()
    closed_eyes_frame_count = 0
    drowsy_detected = False  # To avoid continuous alarm triggering

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur lors de la capture de l'image depuis la caméra.")
            break

        # Crop the frame to the predefined ROI
        roi_frame = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = roi_frame[y:y+h, x:x+w]
            resized_face = cv2.resize(face_roi, (224, 224))
            normalized_face = resized_face / 255.0
            reshaped_face = np.reshape(normalized_face, (1, 224, 224, 3))

            prediction = model.predict(reshaped_face)
            class_index = np.argmax(prediction)
            classes = ['Closed', 'Open', 'no_yawn', 'yawn']
            predicted_class = classes[class_index]

            state = ''
            if predicted_class == 'Closed':
                closed_eyes_frame_count += 1
                if closed_eyes_frame_count >= fps_counter / 2:  # If eyes are closed in half or more of the frames
                    if closed_eyes_frame_count >= fps_counter:  # If eyes are closed in all frames
                        state = 'endormi !'
                    else:
                        state = 'fatigue !'
                    if not drowsy_detected:
                        play_alarm()
                        drowsy_detected = True
            else:
                closed_eyes_frame_count = 0
                state = 'eveille'
                drowsy_detected = False

            # Adjust the coordinates to draw on the original frame
            x += roi_x
            y += roi_y
            cv2.putText(frame, f'Prediction: {state}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        # Increment the frame counter
        fps_counter += 1

        # Calculate and print frames per second every 1 second
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps = fps_counter / elapsed_time
            print(f"Images par seconde : {fps:.2f}")
            print(f"closed_eyes_frame_count: {closed_eyes_frame_count}")  # Debug information
            start_time = time.time()
            fps_counter = 0

    # Libérer la capture vidéo et fermer la fenêtre OpenCV
    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
