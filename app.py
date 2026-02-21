import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
from deepface import DeepFace
import av # PyAV library needed for frame handling

# Page Config
st.set_page_config(page_title="Face Expression Analyzer", layout="wide")

st.title("😃 Facial Expression Analyzer - Live Web App")
st.write("AI Powered Emotion Detection using Deep Learning (WebRTC Mode)")

# DeepFace can be slow on first run, so we wrap it in a function
def analyze_emotion(frame):
    try:
        # DeepFace.analyze returns a list of dictionaries
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        return results[0]['dominant_emotion']
    except Exception as e:
        return None

class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame):
        # Convert the frame from the webcam into an ndarray (BGR for OpenCV)
        img = frame.to_ndarray(format="bgr24")

        # Process emotion
        # Note: We use the image directly. DeepFace handles RGB/BGR internally if needed
        emotion = analyze_emotion(img)

        if emotion:
            # Add text overlay to the video stream
            cv2.putText(img, emotion.upper(), (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        # Return the processed frame back to the browser
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# This creates the UI for the webcam
ctx = webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=EmotionProcessor,
    rtc_configuration={ # This helps connection through firewalls
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)

st.sidebar.info("Click 'Start' to activate your camera. The AI will detect your dominant emotion in real-time.")
