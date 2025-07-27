<<<<<<< HEAD
from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
from demo import calculate_finger_positions, get_finger_states, recognize_letter

app = Flask(__name__)

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

current_letter = ""
text_history = []

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                points = calculate_finger_positions(hand_landmarks)
                finger_states = get_finger_states(points)
                letter = recognize_letter(finger_states, points)
                
                if letter != '?':
                    global current_letter, text_history
                    current_letter = letter
                    if not text_history or text_history[-1] != letter:
                        text_history.append(letter)
                        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/portfolio')
def portfolio():
    return render_template('portfolio.html')

# Fix: Remove the 'i' before @app.route
@app.route('/get_text')
def get_text():
    return {
        'current_letter': current_letter,
        'word': ''.join(text_history)
    }

@app.route('/clear_text')
def clear_text():
    global current_letter, text_history
    current_letter = ""
    text_history = []
    return {'status': 'success'}

if __name__ == "__main__":
=======
from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
from demo import calculate_finger_positions, get_finger_states, recognize_letter

app = Flask(__name__)

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

current_letter = ""
text_history = []

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                points = calculate_finger_positions(hand_landmarks)
                finger_states = get_finger_states(points)
                letter = recognize_letter(finger_states, points)
                
                if letter != '?':
                    global current_letter, text_history
                    current_letter = letter
                    if not text_history or text_history[-1] != letter:
                        text_history.append(letter)
                        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/portfolio')
def portfolio():
    return render_template('portfolio.html')

# Fix: Remove the 'i' before @app.route
@app.route('/get_text')
def get_text():
    return {
        'current_letter': current_letter,
        'word': ''.join(text_history)
    }

@app.route('/clear_text')
def clear_text():
    global current_letter, text_history
    current_letter = ""
    text_history = []
    return {'status': 'success'}

if __name__ == "__main__":
>>>>>>> 210163bdec435feb0b05ff6e6949204cc76920b0
    app.run(debug=True)