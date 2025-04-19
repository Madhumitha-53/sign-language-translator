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
    st.title("Welcome to Sign Language Letter Translator")
    st.markdown("""
    ### Click **Translator** to translate your gestures
    ### Click **Portfolio** to know about our team
    """)

elif page == "Translator":
    st.title("Sign Language Translator")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        video_feed = st.empty()
    
    with col2:
        current_letter = st.empty()
        word_display = st.empty()
        if st.button("Clear Text", key="translator_clear"):
            st.session_state.text_history = []
            current_letter.empty()
            word_display.empty()
            st.success("Text cleared successfully!")
    
    process_frame()

elif page == "Portfolio":
    st.title("Our Team")
    
    # Team Members
    st.header("Team Members")
    
    team_members = [
        "Madhumitha",
        "Logitha",
        "Layashree",
        "Lavanya",
        "Monika",
        "Harsshanna"
    ]
    
    for member in team_members:
        st.subheader(f"👩‍💻 {member}")
    
    # Project Description
    st.header("About the Project")
    st.write("""
    The Sign Language Translator is an innovative application that uses computer vision 
    and machine learning to translate sign language gestures into text in real-time. 
    This project aims to bridge communication gaps and make sign language more accessible 
    to everyone.
    """)
    
    # Technologies Used
    st.header("Technologies Used")
    st.write("""
    - Computer Vision (OpenCV)
    - Machine Learning (MediaPipe)
    - Web Development (Streamlit)
    - Python Programming
    """)

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
    .stRadio > label {
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

def process_frame():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Unable to access webcam. Please check your camera connection.")
            return
            
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam")
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
                        current_letter.write(f"Current Letter: {letter}")
                        if not st.session_state.text_history or st.session_state.text_history[-1] != letter:
                            st.session_state.text_history.append(letter)
                            word_display.write(f"Word: {''.join(st.session_state.text_history)}")

            video_feed.image(frame, channels="RGB")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        if 'cap' in locals():
            cap.release()

if __name__ == "__main__":
    if page in ["Home", "Translator"]:
        process_frame()