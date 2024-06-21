import streamlit as st
import cv2
import requests
import os
import numpy as np

# URLs of the haarcascade XML files
cascades = {
    'face': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml',
    'eye': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml',
    'smile': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_smile.xml'
}

# Download the XML files if they do not already exist
for name, url in cascades.items():
    xml_filename = f'haarcascade_{name}.xml'
    if not os.path.isfile(xml_filename):
        response = requests.get(url)
        with open(xml_filename, 'wb') as file:
            file.write(response.content)

# Load the Haarcascade XML files
face_cascade = cv2.CascadeClassifier('haarcascade_face.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Test if the cascade files were loaded successfully
if face_cascade.empty() or eye_cascade.empty() or smile_cascade.empty():
    raise IOError('Failed to load one or more cascade classifier xml files')


# Define a function to detect faces, eyes, and smiles in a frame
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

    return frame


# Streamlit app
st.title("Face, Eye, and Smile Detection App")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About", "Detection"])

if page == "Home":
    st.header("Welcome to the Face, Eye, and Smile Detection App!")
    st.write("Use the navigation menu to explore the app.")

elif page == "About":
    st.header("About")
    st.write(
        "This app uses OpenCV's pre-trained Haarcascade classifiers to detect faces, eyes, and smiles in live camera feed.")

elif page == "Detection":
    st.header("Face, Eye, and Smile Detection")
    st.write(
        "The app will use your camera to detect faces, eyes, and smiles in real-time. Click the button below to start the detection.")

    if st.button("Start Camera"):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detected_frame = detect_faces(frame)

            # Convert the frame to RGB
            detected_frame_rgb = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)

            # Display the frame using Streamlit
            st.image(detected_frame_rgb, channels="RGB", use_column_width=True)

            # Add a stop button
            if st.button("Stop Camera"):
                break

        cap.release()

# Optionally, remove the XML files after use
for name in cascades.keys():
    os.remove(f'haarcascade_{name}.xml')