import cv2
import numpy as np
import os
import time

# Define some constants
actions = ['hello', 'thanks', 'yes', 'no']  # Your desired actions

# Define separate parameters for sequences and frames per sequence
num_sequences = 500      # Number of sequences to capture per action
frames_per_sequence = 10 # Number of frames per sequence
output_dir = 'dataset'   # Directory to save collected data

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initial delay to give the user time to prepare
print("Starting data collection in 5 seconds...")
time.sleep(5)

# Start collecting data
for action in actions:
    print(f'Starting collection for action: {action}')
    action_dir = os.path.join(output_dir, action)
    if not os.path.exists(action_dir):
        os.makedirs(action_dir)
    
    for sequence in range(num_sequences):
        print(f'Collecting sequence {sequence} for action "{action}"...')
        # Capture frames for the current sequence
        for frame_num in range(frames_per_sequence):
            ret, image = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Display recording information on the frame
            cv2.putText(image, f'Recording "{action}" - Seq {sequence}, Frame {frame_num}',
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Show the frame in a window
            cv2.imshow('Recording Window', image)

            # Save the frame to the appropriate directory
            frame_path = os.path.join(action_dir, f'{sequence}_{frame_num}.jpg')
            cv2.imwrite(frame_path, image)

            # Use a minimal wait time (1ms) between frames
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # No extra delay between sequences—immediately continue to the next sequence

cap.release()
cv2.destroyAllWindows()

print("Data collection complete.")
