import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model


# Load pre-trained models
emotion_model = load_model("C:\\Capstone Project\\Integrated project\\Emotion-Detection-FER2013-master\\Emotion-Detection-FER2013-master\\emotion_detection_model.h5")
asl_model = load_model("C:\\Capstone Project\\Integrated project\\asl_model.h5")  # ASL model
 # ASL recognition model
asl_labels = ['A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']  # Modify according to your classes


# Mediapipe setup
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)


# Sentence buffer
sentence = []

def predict_emotion(face_image):
    face_image = cv2.resize(face_image, (48, 48))  # Adjust according to your emotion model input size
    face_image = np.expand_dims(face_image, axis=0)
    face_image = face_image / 255.0  # Normalize
    emotion_prediction = emotion_model.predict(face_image)
    emotion_label = np.argmax(emotion_prediction)
    emotions = ["Happy", "Sad", "Angry", "Surprised", "Neutral","Disgusted","Fearful"]  # Modify as per your model
    return emotions[emotion_label]

def predict_asl(hand_image):
    hand_image = cv2.resize(hand_image, (64, 64))  # Adjust according to your ASL model input size
    hand_image = np.expand_dims(hand_image, axis=0)
    hand_image = hand_image / 255.0  # Normalize
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

    # Detect face
    face_results = face_detection.process(rgb_frame)
    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, w_b, h_b = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            face_roi = frame[y:y + h_b, x:x + w_b]
            emotion = predict_emotion(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY))
            cv2.rectangle(frame, (x, y), (x + w_b, y + h_b), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Detect hand
    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Draw hand landmarks
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract bounding box
            x_min = min([lm.x for lm in hand_landmarks.landmark])
            y_min = min([lm.y for lm in hand_landmarks.landmark])
            x_max = max([lm.x for lm in hand_landmarks.landmark])
            y_max = max([lm.y for lm in hand_landmarks.landmark])
            
            x, y, w_b, h_b = int(x_min * w), int(y_min * h), int((x_max - x_min) * w), int((y_max - y_min) * h)
            hand_roi = frame[y:y + h_b, x:x + w_b]
            
            try:
                gesture = predict_asl(cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB))
                sentence.append(gesture)
                cv2.rectangle(frame, (x, y), (x + w_b, y + h_b), (0, 255, 0), 2)
                cv2.putText(frame, gesture, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except Exception as e:
                continue
    
    # Display sentence
    if len(sentence) > 5:  # Arbitrary sentence framing logic
        framed_sentence = " ".join(sentence)
        cv2.putText(frame, framed_sentence, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        sentence = []

    # Show video feed
    cv2.imshow("Emotion and ASL Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()