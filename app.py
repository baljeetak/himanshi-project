import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image

st.set_page_config(page_title="Face Expression Analyzer", layout="wide")

st.title("😃 Facial Expression Analyzer - Live Web App")
st.write("AI Powered Emotion Detection using Deep Learning")

run = st.checkbox("Start Webcam")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.warning("Failed to access webcam")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        result = DeepFace.analyze(
            rgb_frame, 
            actions=['emotion'], 
            enforce_detection=False
        )
        emotion = result[0]['dominant_emotion']
        cv2.putText(frame, emotion.upper(), (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    except:
        pass

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

camera.release()