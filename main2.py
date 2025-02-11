import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import time

# -------------------------------
# Load the Trained LSTM Model
# -------------------------------
# This model was trained on sequences of 10 frames, where each frame is represented by a 126-dim vector
model = load_model('mediapipe_hands_model.h5')
actions = ['hello', 'thanks', 'yes', 'no']  # Class labels

# -------------------------------
# Initialize MediaPipe Hands
# -------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

def extract_hand_keypoints(image):
    """
    Given an RGB image (numpy array), this function:
      - Ensures the image is of type uint8 (MediaPipe expects uint8 images with pixel values 0-255).
      - Processes the image with MediaPipe Hands.
      - For each detected hand, extracts the 21 landmarks (each with x, y, z).
      - If fewer than 21 landmarks are detected for a hand, pads with zeros so that each hand yields a 63-dimensional vector.
      - If only one hand is detected (or none), pads the missing hand(s) with zeros.
      - Returns a concatenated vector of length 126.
    """
    # Ensure the image is uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    results = hands.process(image)
    keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            # If fewer than 21 landmarks, pad with zeros (21*3 = 63 values)
            if len(landmarks) < 63:
                landmarks.extend([0] * (63 - len(landmarks)))
            keypoints.append(np.array(landmarks))
        # If only one hand detected, pad with zeros for the missing hand.
        if len(keypoints) < 2:
            keypoints.append(np.zeros(63))
    else:
        # No hands detected: return zeros for both hands.
        keypoints = [np.zeros(63), np.zeros(63)]
    
    # Concatenate keypoints from both hands to obtain a 126-dimensional vector
    return np.concatenate(keypoints)

# -------------------------------
# Streamlit App: Real-Time Sign Language Detection
# -------------------------------
st.title("Real-Time Sign Language Detection")
st.text("Press 'Start Webcam' to begin.")

start_button = st.button("Start Webcam")

# Create a placeholder for video frames
frame_placeholder = st.empty()

if start_button:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not access the webcam.")
    else:
        # Variables for processing sequences
        sequence_length = 10  # Number of frames per sequence
        keypoint_sequence = []  # Will store keypoint vectors for each frame
        last_prediction_time = time.time()
        last_action = "..."
        
        # Run loop until user stops
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam.")
                break

            # Convert frame from BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Extract keypoints from the frame (will always be a 126-dim vector)
            keypoints = extract_hand_keypoints(rgb_frame)
            keypoint_sequence.append(keypoints)
            
            # If we have a full sequence, perform prediction every 2 seconds to smooth output
            if len(keypoint_sequence) == sequence_length:
                input_sequence = np.expand_dims(np.array(keypoint_sequence), axis=0)  # Shape: (1, 10, 126)
                if time.time() - last_prediction_time > 2:
                    prediction = model.predict(input_sequence)
                    predicted_index = np.argmax(prediction)
                    last_action = actions[predicted_index]
                    last_prediction_time = time.time()
                keypoint_sequence = []  # Reset sequence after prediction
            
            # Overlay the prediction on the frame
            cv2.putText(frame, f'Action: {last_action}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Update the Streamlit display (convert frame from BGR to RGB)
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            
            # Exit loop if 'q' is pressed (this may not work perfectly in Streamlit; close the app to stop)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        hands.close()
        st.success("Webcam stopped.")
