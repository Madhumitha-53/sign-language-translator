from flask import Flask, render_template, Response, redirect, url_for
import cv2
import mediapipe as mp
import numpy as np
from untiled import calculate_finger_positions, get_finger_states, recognize_letter

app = Flask(__name__)

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize the webcam
# Add this at the top with other imports
import atexit

# Modify the camera initialization
def get_camera():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera = cv2.VideoCapture(0)  # Try one more time
    return camera

camera = get_camera()

def generate_frames():
    global camera
    if not camera.isOpened():
        camera = get_camera()
    
    text_history = []
    last_prediction = ''
    prediction_counter = 0
    stable_frames = 8
    
    while True:
        ret, frame = camera.read()
        if not ret:
            camera = get_camera()  # Reinitialize if frame capture fails
            continue
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                points = calculate_finger_positions(hand_landmarks)
                finger_states = get_finger_states(points)
                current_prediction = recognize_letter(finger_states, points)
                
                if current_prediction == last_prediction:
                    prediction_counter += 1
                else:
                    prediction_counter = 0
                
                last_prediction = current_prediction
                
                if prediction_counter >= stable_frames:
                    if current_prediction != '?' and (not text_history or text_history[-1] != current_prediction):
                        text_history.append(current_prediction)
                        if len(text_history) > 10:
                            text_history.pop(0)
                    
                    cv2.putText(frame, f"Current: {current_prediction}", (10, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    word = ''.join(text_history)
                    cv2.putText(frame, f"Word: {word}", (10, 90),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
@app.route('/translator')
def index():
    return render_template('translator.html')

@app.route('/team')
def team():
    return render_template('portfolio.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Add URL routes
@app.route('/sign-to-text')
@app.route('/letter-translator')
@app.route('/hand-signs')
@app.route('/asl-translator')
def redirect_to_translator():
    return redirect('/translator')

# Custom domain suggestions (you'll need to register these):
# signlanguagetech.com
# handtranslator.tech
# signtotext.app
# lettersigner.com
# aslhelper.tech

@app.route('/release_camera', methods=['POST'])
def release_camera():
    global camera
    if camera and camera.isOpened():
        camera.release()
        cv2.destroyAllWindows()
    return '', 204

def cleanup():
    global camera
    if camera and camera.isOpened():
        camera.release()
        cv2.destroyAllWindows()

atexit.register(cleanup)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)