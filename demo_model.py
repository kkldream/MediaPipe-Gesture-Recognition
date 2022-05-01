import cv2
import mediapipe as mp
import pandas as pd
from keras.models import load_model
import numpy as np
import utils
import os
from PIL import Image, ImageDraw, ImageFont

# model = load_model('model.hdf5')
# model = load_model('model_dense.hdf5')  # overfitting
# model = load_model('model_conv1d.hdf5')  # good
model = load_model('model_conv2d.hdf5')  # double good

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75)

labels = os.listdir(os.path.join('dataset', 'markers'))
print('labels:')
for i, label in enumerate(labels):
    print(f'{i}. {label}')


def main():
    cap = cv2.VideoCapture(0)
    cvFpsCalc = utils.CvFpsCalc(buffer_len=10)
    loop = 0
    while True:
        display_fps = cvFpsCalc.get()
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        results = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # if results.multi_handedness:
        #     for hand_label in results.multi_handedness:
        #         print(hand_label)
        y_data = []
        x_list = []
        y_list = []
        z_list = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x_list.append(landmark.x)
                    y_list.append(landmark.y)
                    z_list.append(landmark.z)
                # print('hand_landmarks:', hand_landmarks)
                # 关键点可视化
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            data = pd.DataFrame({
                'x': x_list,
                'y': y_list,
                'z': z_list,
            })
            if data.shape == (21, 3):
                data = np.array(data)
                data = data[:, :, np.newaxis]
                # print(data.head(5))
                # print(data.shape)
                predicted = model.predict(np.array([data]))

                max_var = (0, 0);
                for i, var in enumerate(predicted[0]):
                    print(f'{int(var * 100):3}%, ', end='')
                    if var > max_var[1]:
                        max_var = (i, var)
                print(labels[max_var[0]])
                if max_var[1] > 0.5:
                    cv2.putText(frame, f'{max_var[1]}%', (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    frame = cv2AddChineseText(frame, f'{labels[max_var[0]]}', (10, 90), textColor=(0, 0, 255),
                                              textSize=30)
                # cv2.putText(frame, f'{labels[max_var[0]]}', (10, 90),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # path = f'data/{3}/{loop}.csv'
                # data.to_csv(path)
        # hand_landmarks_list = cul_mk_list(results)
        #
        # if len(hand_landmarks_list) > 0:
        #     print(hand_landmarks_list[0])
        ''' display '''
        cv2.putText(frame, f'FPS: {display_fps}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('MediaPipe Hands', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        loop += 1
    cap.release()


def cul_mk_list(results):
    hand_landmarks_list = []
    if not results.multi_hand_landmarks:
        return hand_landmarks_list
    for hand_landmarks in results.multi_hand_landmarks:
        for landmark in hand_landmarks.landmark:
            hand_landmarks_list.append([landmark.x, landmark.y, landmark.z])
    return hand_landmarks_list


def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


if __name__ == '__main__':
    main()
