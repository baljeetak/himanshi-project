import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import cv2
import av
import numpy as np
from deepface import DeepFace
import logging

# Set up page headers
st.set_page_config(page_title="Face Expression AI", layout="wide")
st.title("😃 Facial Expression Analyzer")
st.write("Real-time Emotion Detection using DeepFace & WebRTC")

# Configure WebRTC to punch through firewalls (STUN Servers)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    try:
        # Analyze the frame for emotions
        # enforce_detection=False prevents crashing when no face is visible
        results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        
        # Get the dominant emotion
        emotion = results[0]['dominant_emotion']
        
        # Draw the emotion text on the image
        cv2.putText(img, emotion.upper(), (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    except Exception as e:
        # If detection fails, just return the original frame
        pass

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# UI Layout
col1, col2 = st.columns([2, 1])

with col1:
    webrtc_streamer(
        key="emotion-detection",
        video_frame_callback=video_frame_callback,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("Instructions")
    st.markdown("""
    1. Click **Start** to open your webcam.
    2. Ensure your face is well-lit.
    3. The AI will label your dominant emotion (Happy, Sad, Angry, etc.) directly on the video.
    """)
    
    st.warning("Note: The first frame may take a moment to process as the AI models load.")
