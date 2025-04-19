import streamlit as st
import cv2
import mediapipe as mp
from demo import calculate_finger_positions, get_finger_states, recognize_letter

st.set_page_config(page_title="Sign Language Translator", layout="wide")

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize session state
if 'text_history' not in st.session_state:
    st.session_state.text_history = []
    st.session_state.last_prediction = ''
    st.session_state.prediction_counter = 0

# Navigation
page = st.radio("Navigation", ["Home", "Translator", "Portfolio"], horizontal=True, label_visibility="hidden")

if page == "Home":
    # Home page content remains the same
    ...

elif page == "Translator":
    st.title("Sign Language Translator")
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])

    with col1:
        video_feed = st.empty()

    with col2:
        current_letter = st.empty()
        word_display = st.empty()
        
        if st.button("Clear Text"):
            st.session_state.text_history = []
            current_letter.empty()
            word_display.empty()
            st.success("Text cleared successfully!")

    process_frame()

elif page == "Portfolio":
    # Portfolio page content remains the same
    ...

# Add styling
st.markdown("""
    <style>
    div.stButton > button {
        background-color: #3498db;
        color: white;
        font-size: 20px;
        padding: 20px;
        border-radius: 10px;
    }
    div.stButton > button:hover {
        background-color: #2980b9;
    }
    </style>
    """, unsafe_allow_html=True)

def process_frame():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

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
                    current_letter.write(f"Current Letter: {letter}")
                    if not st.session_state.text_history or st.session_state.text_history[-1] != letter:
                        st.session_state.text_history.append(letter)
                        word_display.write(f"Word: {''.join(st.session_state.text_history)}")

        video_feed.image(frame, channels="RGB")

if __name__ == "__main__":
    process_frame()