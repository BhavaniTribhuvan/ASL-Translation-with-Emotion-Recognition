import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time

# Load pre-trained models

asl_model = load_model("C:\\Capstone Project\\Integrated project\\asl_model.h5")  # ASL model
asl_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# Sentence buffer
sentence = []
last_added_time = time.time()

# ASL recognition function
def predict_asl(hand_image):
    hand_image = cv2.resize(hand_image, (28, 28))  # Resize to match model input size
    hand_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    hand_image = hand_image.astype('float32') / 255.0  # Normalize to [0, 1]
    hand_image = np.expand_dims(hand_image, axis=0)  # Add batch dimension
    hand_image = np.expand_dims(hand_image, axis=-1)  # Add channel dimension
    asl_prediction = asl_model.predict(hand_image)
    asl_label = np.argmax(asl_prediction)
    return asl_labels[asl_label]


# Webcam feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for natural interaction
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect hand
    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract bounding box coordinates for the hand region
            x_min = min([lm.x for lm in hand_landmarks.landmark])
            y_min = min([lm.y for lm in hand_landmarks.landmark])
            x_max = max([lm.x for lm in hand_landmarks.landmark])
            y_max = max([lm.y for lm in hand_landmarks.landmark])

            x, y, w_b, h_b = int(x_min * w), int(y_min * h), int((x_max - x_min) * w), int((y_max - y_min) * h)
            hand_roi = frame[y:y + h_b, x:x + w_b]

            try:
                gesture = predict_asl(cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB))
                if time.time() - last_added_time > 3:  # Add letter only after 1 second
                    if len(sentence) == 0 or (gesture != sentence[-1]):
                        sentence.append(gesture)
                        last_added_time = time.time()

                cv2.rectangle(frame, (x, y), (x + w_b, y + h_b), (0, 255, 0), 2)
                cv2.putText(frame, gesture, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            except Exception as e:
                print(f"Error in ASL detection: {e}")
                # Display sentence
    if len(sentence) > 0:
        framed_sentence = " ".join(sentence)
        cv2.putText(frame, framed_sentence, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show video feed
    cv2.imshow("ASL Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    
    

