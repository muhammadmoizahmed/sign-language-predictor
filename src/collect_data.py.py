import cv2
import mediapipe as mp
import pandas as pd
import os

# Sign name input lo
sign_name = input("Which hand sign would you like to capture?(eg: A,B,1,2): ")

# Mediapipe Hands init
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Camera open karo
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
2
landmarks_list = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for selfie-view
    frame = cv2.flip(frame, 1)

    # RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.y)

            landmarks.insert(0, sign_name)  # Sign name at start
            landmarks_list.append(landmarks)

            cv2.putText(frame, f"Captured: {sign_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Capture", frame)

    # Q dabao to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# DataFrame banayo
columns = ['class'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)]
df = pd.DataFrame(landmarks_list, columns=columns)

# Agar file hai → append karo warna naye se banao
if os.path.exists('sign_data.csv'):
    df.to_csv('sign_data.csv', mode='a', header=False, index=False)
    print(" New data appended to sign_data.csv")
else:
    df.to_csv('sign_data.csv', index=False)
    print(" New file sign_data.csv created")

print(f"Captured {len(landmarks_list)} samples for: {sign_name}")
