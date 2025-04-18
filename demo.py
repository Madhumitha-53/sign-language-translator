import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import string
import math
from collections import deque
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

def calculate_finger_positions(hand_landmarks):
    finger_points = []
    for i in range(21):  # MediaPipe hand has 21 points
        point = hand_landmarks.landmark[i]
        finger_points.append([point.x, point.y, point.z])
    return np.array(finger_points)

def get_finger_states(points):
    finger_states = []
    
    # Thumb detection
    thumb_tip = points[4]
    thumb_base = points[2]
    thumb_extended = thumb_tip[0] < thumb_base[0]
    finger_states.append(thumb_extended)
    
    # Finger detection
    for finger in range(4):
        tip_id = 8 + (finger * 4)
        mid_id = 7 + (finger * 4)
        base_id = 5 + (finger * 4)
        
        finger_tip = points[tip_id]
        finger_mid = points[mid_id]
        finger_base = points[base_id]
        
        vertical_check = finger_tip[1] < finger_base[1]
        distance_check = np.linalg.norm(finger_tip - finger_base) > 0.1
        
        finger_extended = vertical_check and distance_check
        finger_states.append(finger_extended)
    
    return finger_states

def recognize_letter(finger_states, points):
    patterns = {
        'A': [False, False, False, False, False],    # Fist
        'B': [True, True, True, True, True],         # Flat hand
        'C': [True, True, True, True, True],         # Curved C
        'D': [False, True, False, False, False],     # Index up
        'E': [True, False, False, False, False],     # Thumb
        'F': [True, True, False, False, False],      # First two
        'G': [True, True, False, False, False],      # Point sideways
        'H': [False, True, True, False, False],      # Index and middle
        'I': [False, False, False, False, True],     # Pinky
        'J': [False, False, False, False, True],     # Moving pinky
        'K': [True, True, False, False, False],      # Victory pointing
        'L': [True, True, False, False, False],      # L shape
        'M': [False, True, True, True, False],       # Three down
        'N': [False, True, True, False, False],      # Two down
        'O': [True, True, True, True, True],         # O shape
        'P': [True, True, False, False, False],      # P shape
        'Q': [True, True, False, False, False],      # Q shape
        'R': [False, True, True, False, False],      # Crossed fingers
        'S': [False, False, False, False, False],    # Fist
        'T': [False, True, False, False, False],     # T shape
        'U': [False, True, True, False, False],      # Two up close
        'V': [False, True, True, False, False],      # Two up spread
        'W': [False, True, True, True, False],       # Three up
        'X': [False, True, False, False, False],     # Bent index
        'Y': [True, False, False, False, True],      # Hang loose
        'Z': [False, True, False, False, False],     # Moving index
    }

    palm_center = np.mean(points[5:9], axis=0)
    
    for symbol, pattern in patterns.items():
        if finger_states == pattern:
            if symbol in ['U', 'V']:
                finger_spread = np.linalg.norm(points[8] - points[12])
                return 'V' if finger_spread > 0.13 else 'U'
            
            elif symbol in ['M', 'N', 'W']:
                finger_spread = np.linalg.norm(points[8] - points[16])
                if finger_spread > 0.15:
                    return 'W'
                return 'M' if finger_states[3] else 'N'
            
            elif symbol in ['A', 'S', 'E']:
                thumb_pos = points[4][0] - points[2][0]
                if thumb_pos > 0.05:
                    return 'E'
                return 'S' if points[4][1] > points[2][1] else 'A'
            
            elif symbol in ['K', 'P', 'R']:
                angle = np.arctan2(points[8][1] - points[5][1], 
                                 points[8][0] - points[5][0])
                if abs(angle) > 1.2:
                    return 'K'
                return 'R' if abs(angle) < 0.3 else 'P'
            
            return symbol
    
    return '?'

# Add accuracy tracking
CONFIDENCE_THRESHOLD = 0.85
PREDICTION_HISTORY_SIZE = 30

class SignLanguageGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("Sign Language Translator")
        
        # Add these lines
        self.last_prediction = ''
        self.prediction_counter = 0
        self.text_history = []
        self.words = []
        self.last_letter_time = time.time()
        self.prediction_history = []
        
        # Create frames
        self.video_frame = ttk.Frame(window)
        self.video_frame.grid(row=0, column=0, padx=10, pady=5)
        
        self.text_frame = ttk.Frame(window)
        self.text_frame.grid(row=0, column=1, padx=10, pady=5)
        
        # Video canvas
        self.canvas = tk.Canvas(self.video_frame, width=640, height=480)
        self.canvas.pack()
        
        # Text displays
        self.letter_label = ttk.Label(self.text_frame, text="Current Letter: ", font=('Arial', 14))
        self.letter_label.pack(pady=5)
        
        self.word_label = ttk.Label(self.text_frame, text="Current Word: ", font=('Arial', 14))
        self.word_label.pack(pady=5)
        
        self.sentence_label = ttk.Label(self.text_frame, text="Sentence: ", font=('Arial', 14), wraplength=300)
        self.sentence_label.pack(pady=5)
        
        self.accuracy_label = ttk.Label(self.text_frame, text="Accuracy: 0%", font=('Arial', 14))
        self.accuracy_label.pack(pady=5)
        
        # Control buttons
        self.clear_button = ttk.Button(self.text_frame, text="Clear (C)", command=self.clear_text)
        self.clear_button.pack(pady=5)
        
        self.quit_button = ttk.Button(self.text_frame, text="Quit (Q)", command=self.quit_app)
        self.quit_button.pack(pady=5)
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.update()
    
    def clear_text(self):
        global text_history, words
        text_history.clear()
        words.clear()
        self.word_label.config(text="Current Word: ")
        self.sentence_label.config(text="Sentence: ")
    
    def quit_app(self):
        self.cap.release()
        self.window.quit()
    
    def calculate_accuracy(self, prediction_history):
        if not prediction_history:
            return 0
        
        total = len(prediction_history)
        matches = sum(1 for i in range(1, total) 
                     if prediction_history[i] == prediction_history[i-1])
        return (matches / (total - 1)) * 100 if total > 1 else 0
    
    def update(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            current_prediction = '?'
            current_word = ''
            sentence = ''
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    points = calculate_finger_positions(hand_landmarks)
                    finger_states = get_finger_states(points)
                    current_prediction = recognize_letter(finger_states, points)
                    
                    if current_prediction == self.last_prediction:
                        self.prediction_counter += 1
                    else:
                        self.prediction_counter = 0
                    
                    self.last_prediction = current_prediction
                    
                    if self.prediction_counter >= 8:  # stable_frames
                        current_time = time.time()
                        
                        if current_prediction != '?' and (not self.text_history or self.text_history[-1] != current_prediction):
                            self.last_letter_time = current_time
                            self.text_history.append(current_prediction)
                            if len(self.text_history) > 15:
                                self.text_history.pop(0)
                        
                        if current_time - self.last_letter_time > 2.0:  # WORD_TIMEOUT
                            word = process_text_history(self.text_history)
                            if word and (not self.words or self.words[-1] != word):
                                self.words.append(word)
                                self.text_history.clear()
                                if len(self.words) > 5:
                                    self.words.pop(0)
                    
                    current_word = ''.join(self.text_history)
                    sentence = ' '.join(self.words + ([current_word] if current_word else []))
            
            # Update GUI elements
            self.letter_label.config(text=f"Current Letter: {current_prediction}")
            self.word_label.config(text=f"Current Word: {current_word}")
            self.sentence_label.config(text=f"Sentence: {sentence}")
            
            # Update prediction history and accuracy
            self.prediction_history.append(current_prediction)
            if len(self.prediction_history) > PREDICTION_HISTORY_SIZE:
                self.prediction_history.pop(0)
            accuracy = self.calculate_accuracy(self.prediction_history)
            self.accuracy_label.config(text=f"Accuracy: {accuracy:.1f}%")
            
            # Display frame
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(display_frame))
            self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            self.canvas.photo = photo
        
        self.window.after(10, self.update)

# Modify main() to only handle GUI
def main():
    root = tk.Tk()
    app = SignLanguageGUI(root)
    root.mainloop()
    last_prediction = ''
    prediction_counter = 0
    stable_frames = 8
    text_history = []
    words = []
    last_letter_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        text_display = np.zeros((200, 600, 3), dtype=np.uint8)
        
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
                    current_time = time.time()
                    
                    if current_prediction != '?' and (not text_history or text_history[-1] != current_prediction):
                        # Reset timer for new letter
                        last_letter_time = current_time
                        text_history.append(current_prediction)
                        if len(text_history) > 15:  # Increased buffer for words
                            text_history.pop(0)
                    
                    # Check for word completion
                    if current_time - last_letter_time > WORD_TIMEOUT and text_history:
                        word = process_text_history(text_history)
                        if word and (not words or words[-1] != word):
                            words.append(word)
                            text_history.clear()
                            if len(words) > 5:  # Keep last 5 words
                                words.pop(0)
                    
                    # Display current letter and word
                    cv2.putText(frame, f"Letter: {current_prediction}", (10, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    current_word = ''.join(text_history)
                    cv2.putText(frame, f"Current: {current_word}", (10, 90),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Display sentence
                    sentence = ' '.join(words + ([current_word] if current_word else []))
                    cv2.putText(text_display, f"Sentence: {sentence}", (10, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Sign Language Translator', frame)
        cv2.imshow('Detected Text', text_display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            text_history.clear()
            words.clear()

    cap.release()
    cv2.destroyAllWindows()

# Add these word mappings after the imports
COMMON_WORDS = {
    # Greetings
    'HELLO': 'HELLO',
    'HI': 'HI',
    'BYE': 'GOODBYE',
    'GM': 'GOOD MORNING',
    'GN': 'GOOD NIGHT',
    
    # Common Phrases
    'PLZ': 'PLEASE',
    'THX': 'THANK YOU',
    'SRY': 'I AM SORRY',
    'NP': 'NO PROBLEM',
    'OK': 'OKAY',
    
    # Questions
    'HOW': 'HOW ARE YOU',
    'WHAT': 'WHAT IS',
    'WHERE': 'WHERE IS',
    'WHEN': 'WHEN IS',
    'WHO': 'WHO IS',
    
    # Responses
    'FINE': 'I AM FINE',
    'YES': 'YES',
    'NO': 'NO',
    'IDK': 'I DO NOT KNOW',
    
    # Needs
    'HELP': 'I NEED HELP',
    'FOOD': 'I AM HUNGRY',
    'WATER': 'I AM THIRSTY',
    'REST': 'I AM TIRED',
    'SICK': 'I AM SICK',
    
    # Emotions
    'HAPPY': 'I AM HAPPY',
    'SAD': 'I AM SAD',
    'ANGRY': 'I AM ANGRY',
    'LOVE': 'I LOVE YOU',
    
    # Places
    'HOME': 'AT HOME',
    'SCHOOL': 'AT SCHOOL',
    'WORK': 'AT WORK',
    'HOSP': 'HOSPITAL',
    
    # Emergency
    'HELP': 'I NEED HELP',
    'DOC': 'I NEED A DOCTOR',
    'EMG': 'EMERGENCY',
    'CALL': 'PLEASE CALL',
    
    # Common Actions
    'COME': 'COME HERE',
    'WAIT': 'PLEASE WAIT',
    'STOP': 'STOP',
    'GO': 'GO AHEAD',
    'LOOK': 'LOOK AT THIS'
}

# Add this function to improve word detection
def detect_special_patterns(text_history):
    text = ''.join(text_history)
    
    # Check for number patterns
    if text.isdigit():
        return f"NUMBER {text}"
    
    # Check for time patterns (e.g., 123 -> 1:23)
    if len(text) == 3 and text.isdigit():
        return f"TIME {text[0]}:{text[1:]}"
    
    # Check for common abbreviations
    return COMMON_WORDS.get(text, text)

# Update the process_text_history function
def process_text_history(text_history):
    if not text_history:
        return ""
    
    word = ''.join(text_history)
    
    # Check for special patterns first
    special_word = detect_special_patterns(text_history)
    if special_word != word:
        return special_word
    
    # Check for partial matches in common words
    for key, phrase in COMMON_WORDS.items():
        if key in word:
            return phrase
    
    return word

if __name__ == "__main__":
    main()