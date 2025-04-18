import streamlit as st
from app import app, generate_frames
import cv2
from PIL import Image
import io

st.title("Sign Language Translator")

if st.button("Start Camera"):
    video_placeholder = st.empty()
    translation_placeholder = st.empty()
    
    while True:
        frame = generate_frames()
        if frame:
            video_placeholder.image(frame, channels="BGR")

if st.button("Stop"):
    st.stop()