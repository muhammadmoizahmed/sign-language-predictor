import cv2
import mediapipe as mp
import joblib
import numpy as np
import time
import textwrap

# Load trained model
model = joblib.load('./models/sign_model.pkl')

# Initialize mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

sentence = ""
last_prediction_time = 0
delay = 2.00

prediction_history = []
history_size = 5

# White window for sentence
text_window = np.ones((300, 800, 3), dtype=np.uint8) * 255  # white background

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    current_time = time.time()

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.y)

            X = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(X)[0]

            prediction_history.append(prediction)
            if len(prediction_history) > history_size:
                prediction_history.pop(0)

            final_prediction = max(set(prediction_history), key=prediction_history.count)

            # Show prediction on camera screen
            cv2.putText(frame, f"Predicted: {final_prediction}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            if prediction_history.count(final_prediction) >= 3:
                if (current_time - last_prediction_time) > delay:
                    if final_prediction == "SPACE":
                        sentence += " "
                    elif final_prediction == "BACKSPACE":
                        sentence = sentence[:-1]
                    else:
                        sentence += final_prediction
                    last_prediction_time = current_time

    # Clear text window and draw updated sentence
    text_window[:] = 255  # white background

    wrapped_text = textwrap.wrap(sentence, width=40)
    line_height = 30
    y0 = 50

    for i, line in enumerate(wrapped_text):
        y = y0 + i * line_height
        cv2.putText(text_window, line, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Show windows
    cv2.imshow("Camera", frame)
    cv2.imshow("Text Output", text_window)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
