import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from demo import calculate_finger_positions, get_finger_states, recognize_letter

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

def main():
    st.title("Sign Language Translator")
    
    # Create placeholders
    frame_placeholder = st.empty()
    text_placeholder = st.empty()
    
    text_history = []
    last_prediction = ''
    prediction_counter = 0
    stable_frames = 8

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
                    
                    word = ''.join(text_history)
                    text_placeholder.text(f"Current Letter: {current_prediction}\nWord: {word}")

        frame_placeholder.image(frame, channels="BGR")

    cap.release()

if __name__ == "__main__":
    main()