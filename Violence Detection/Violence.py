import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('MoBiLSTM_model.h5')

# Define your class labels
label_names = ['non-violence', 'violence']  # Modify this if needed

# Parameters
frame_size = (64, 64)  # same as training
sequence_length = 16   # model expects 16 frames per sequence
sequence_buffer = []

# Load the video
video_path = 't2.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and normalize frame
    frame_resized = cv2.resize(frame, frame_size)
    frame_normalized = frame_resized / 255.0  # normalize to [0, 1]
    
    sequence_buffer.append(frame_normalized)

    # Keep only the last `sequence_length` frames
    if len(sequence_buffer) > sequence_length:
        sequence_buffer.pop(0)

    # Predict only when we have enough frames
    if len(sequence_buffer) == sequence_length:
        sequence_input = np.array(sequence_buffer)
        
        # Reshape to match expected input shape: (1, 16, 64, 64, 3)
        sequence_input = sequence_input.reshape(1, sequence_length, 64, 64, 3)

        prediction = model.predict(sequence_input)
        predicted_class = np.argmax(prediction)

        # Change text color based on prediction
        if predicted_class == 1:  # Violence (index 1)
            text_color = (0, 0, 255)  # Red color (BGR)
        else:  # Non-violence (index 0)
            text_color = (0, 255, 0)  # Green color (BGR)

        # Display prediction text with the selected color
        cv2.putText(frame, f'Predicted: {label_names[predicted_class]}', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

    # Display the frame (frame color stays unchanged)
    cv2.imshow('Violence Detection', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
