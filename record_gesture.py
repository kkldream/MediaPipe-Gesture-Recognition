import os
import cv2
import mediapipe as mp
import pandas as pd

# label = 'paper'
# label = 'scissors'
label = 'Stone'

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)


def main():
    cap = cv2.VideoCapture(0)
    num = 0
    markers_path = os.path.join('dataset', 'markers', label)
    if not os.path.exists(markers_path):
        os.makedirs(markers_path)
    else:
        num = len(os.listdir(markers_path))
    images_path = os.path.join('dataset', 'images', label)
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        original_frame = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                x_list, y_list, z_list = [], [], []
                for landmark in hand_landmarks.landmark:
                    x_list.append(landmark.x)
                    y_list.append(landmark.y)
                    z_list.append(landmark.z)
                data = pd.DataFrame({
                    'x': x_list,
                    'y': y_list,
                    'z': z_list,
                })
                data_path = os.path.join(markers_path, f'{num}.csv')
                data.to_csv(data_path)
                num += 1
                print(num)
                image_path = os.path.join(images_path, f'{num}.jpg')
                cv2.imwrite(image_path, original_frame)
                if num >= 500:
                    cap.release()
                    print(f'label = {label}, num = {num + 1}')
                    exit()
        cv2.imshow('MediaPipe Hands', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    print(f'label = {label}, num = {num + 1}')


if __name__ == '__main__':
    main()
