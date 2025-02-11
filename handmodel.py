import os
import cv2
import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import metrics
import mediapipe as mp
import random

# -------------------------------
# Dataset and Preprocessing Setup
# -------------------------------

# Path to your dataset directory
DATASET_PATH = r'C:\Britti\projects\sign language\dataset'

# Define your action labels (folder names in your dataset)
actions = np.array(['hello', 'thanks', 'yes', 'no'])

# Define the number of sequences per action and frames per sequence
sequences = 500   # Number of sequences per action (adjust as needed)
frames = 10      # Number of frames per sequence

# Create a label map (e.g., {'hello': 0, 'thanks': 1, 'yes': 2, 'no': 3})
label_map = {label: num for num, label in enumerate(actions)}

# -------------------------------
# Initialize MediaPipe Hands
# -------------------------------
mp_hands = mp.solutions.hands
# static_image_mode=True since we're processing images, and max_num_hands=2 to capture both hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

def extract_hand_keypoints(image):
    """
    Extracts hand landmarks from an image using MediaPipe Hands.
    Expects an RGB image.
    
    - Ensures the image is of type uint8.
    - Processes the image with MediaPipe Hands.
    - For each detected hand, extracts landmarks.
      * If the detected landmarks count is less than 21 (i.e. 17 keypoints, resulting in 51 values),
        pads the vector with zeros to reach 63.
    - If fewer than 2 hands are detected, pads with zeros.
    Returns a flattened vector of length 126 (63 per hand * 2 hands).
    """
    # Ensure image is uint8 (MediaPipe expects uint8 images with values in 0-255)
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    results = hands.process(image)
    
    keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            # Check if the landmarks vector is shorter than expected (should be 21*3 = 63)
            if len(landmarks) < 63:
                # For example, if only 17 landmarks were returned, pad with zeros:
                landmarks.extend([0] * (63 - len(landmarks)))
            keypoints.append(np.array(landmarks))
        # If only one hand was detected, append zeros for the missing hand.
        if len(keypoints) < 2:
            keypoints.append(np.zeros(63))
    else:
        # No hands detected: return zeros for both hands.
        keypoints = [np.zeros(63), np.zeros(63)]
    
    # Concatenate both hands to form a 126-dimensional vector.
    keypoints = np.concatenate(keypoints)
    return keypoints

# -------------------------------
# (Optional) Data Augmentation Function
# -------------------------------
def augment_image(image):
    """
    Applies random augmentation to an image:
      - With 50% chance, rotates the image between -10 and 10 degrees.
      - With 50% chance, adjusts brightness.
    """
    if random.random() < 0.5:
        angle = random.uniform(-10, 10)
        (h, w) = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
    if random.random() < 0.5:
        factor = random.uniform(0.8, 1.2)
        image = np.clip(image * factor, 0, 255).astype(np.uint8)
    return image

# -------------------------------
# Build the Dataset Using MediaPipe Hands Keypoints (with Augmentation)
# -------------------------------
keypoint_data = []
labels_list = []

for action, sequence in product(actions, range(sequences)):
    temp_sequence = []
    for frame_num in range(frames):
        # Expected file naming: "<sequence>_<frame>.jpg"
        image_path = os.path.join(DATASET_PATH, action, f'{sequence}_{frame_num}.jpg')
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            if image is not None:
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Apply augmentation for classes "yes" and "no"
                if action in ['yes', 'no']:
                    image = augment_image(image)
                # Resize to 224x224 for consistency (the extraction function handles its own resizing if needed)
                image = cv2.resize(image, (224, 224))
                # Normalize to [0,1] as float32
                image = image.astype(np.float32) / 255.0
                # Extract keypoints using MediaPipe Hands
                keypoints = extract_hand_keypoints(image)
                temp_sequence.append(keypoints)
    if len(temp_sequence) == frames:
        keypoint_data.append(temp_sequence)
        labels_list.append(label_map[action])

# Release MediaPipe resources
hands.close()

X = np.array(keypoint_data)  # Shape: (num_samples, frames, 126)
Y = to_categorical(labels_list, num_classes=len(actions))

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.10, random_state=34, stratify=labels_list
)

# -------------------------------
# Model Definition: LSTM on Hand Keypoints
# -------------------------------
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(frames, 126)))
model.add(LSTM(64, return_sequences=True, activation='relu'))
model.add(LSTM(32, return_sequences=False, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

model.fit(X_train, Y_train, epochs=150, batch_size=8, validation_data=(X_test, Y_test), callbacks=[early_stopping])

model.save('mediapipe_hands_model.h5')

predictions_prob = model.predict(X_test)
predictions = np.argmax(predictions_prob, axis=1)
test_labels = np.argmax(Y_test, axis=1)
accuracy = metrics.accuracy_score(test_labels, predictions)
print("Final Accuracy:", accuracy)
