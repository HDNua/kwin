"""
 video recognizer

 Developer: HeeJun Choi, DoYoung Han
 Version: 0.6.5
 Release Date: 2017-01-09
"""

import numpy as np
import cv2
import tensorflow as tf
import sys
from kwin import *
import time
import dataset

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

########################################################################
# 프로그램을 초기화합니다.
########################################################################
# 변수를 선언합니다.
test_video_path = dataset.test_data_path("testing2.mp4")

sess = None                 # 세션에 대한 참조입니다.
softmax_tensor = None       # 세션의 그래프 구조에 대한 참조입니다.

grayscale = False           # 프레임을 회색으로 변환하려면 grayscale을 True로 설정합니다.

moved = False               # 프레임과 출력 창의 시작 위치를 지정하려면 값을 False로 둡니다.
FRAME_X_ORIGIN = 0
FRAME_Y_ORIGIN = 0
BOARD_X_ORIGIN = FRAME_X_ORIGIN + 960
BOARD_Y_ORIGIN = 0

OUT_X_ORIGIN = 50           # x 시작점입니다.
OUT_Y_ORIGIN = 50           # y 시작점입니다.
OUT_LINE_SPACE = 50         # 출력의 줄 간격입니다.
OUT_FONT_SIZE = 1         # 폰트 크기입니다

# 출력 창을 위한 흰색 배경 이미지입니다.
white = cv2.imread("white-640x480.jpg")
if white is None:
    print("cannot read white")
    sys.exit()

# 웹 캠을 엽니다.
video = cv2.VideoCapture(0)
# video = cv2.VideoCapture(test_video_path)

frame_list = []
frame_time = []

# 레이블 파일을 가져오고(GFile) 모든 줄의 끝에 있는 캐리지 리턴을 제거합니다(rstrip).
train_labels_path = dataset.output_labels()
label_lines = [line.rstrip() for line
               in tf.gfile.GFile(train_labels_path)]

# 그래프 파일로부터 그래프를 생성합니다.
train_graph_path = dataset.output_graph()
with tf.gfile.FastGFile(train_graph_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


# 매 프레임을 세는 카운트 및 1프레임 당 시간은 0.041666666666666666667초 ( 1/24초 )
# print (label_lines)
# obj = input("찾고자 하는 물체 입력 : ")
# count = 1
# time = 0.04166666666666666666666666666667
########################################################################
# 파일이 열려있는 동안 작업을 진행합니다.
# !!!!! IMPORTANT !!!!
# Python에서 None과의 비교를 위해 != 연산자를 사용하지 마십시오.
while video is not None and video.isOpened():
    # 영상 파일로부터 데이터를 가져옵니다.
    ret, frame = video.read()
    # count = count + 1
    # if count % 4 != 0:
    #     continue

    if frame is None:
        break

    # 화면에 출력할 텍스트의 리스트를 정의합니다.
    text_list = []

    # 프레임을 회색으로 변환하려면 grayscale을 True로 설정합니다.
    if grayscale is True:
        # 프레임을 흑백으로 변환합니다.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # cv2.waitKey(int delay) -> https://goo.gl/lhjVHH
    # delay 밀리초마다 키 이벤트를 대기합니다.
    ch = cv2.waitKey(1)

    # Q를 누르면 종료합니다.
    if ch == 113:
        break

    # Numpy 행렬 형태로 프레임을 변환합니다.
    #  0:3 -> RGB 값을 유지합니다.
    pil_img = np.array(frame)[:, :, 0:3]

    # softmax_tensor를 단 한 번만 초기화합니다.
    if softmax_tensor is None:
        sess = tf.Session()
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    # 1차 예측의 확률을 획득합니다.
    predictions = sess.run(softmax_tensor, {'DecodeJpeg:0': pil_img})

    # 신뢰도 순서로 1차 예측의 레이블을 보여주기 위해 정렬합니다.
    #  (Sort to show labels of first prediction in order of confidence)
    # https://docs.python.org/2.3/whatsnew/section-slices.html
    # * [start:end:step], start(in)부터 end(ex)까지 step 단위로 가져옵니다.
    # > [::-1] -> 모든 범위의 원소를 거꾸로 정렬합니다.
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    # 정렬된 레이블을 출력합니다.
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]

        # 출력을 보관합니다.
        text_list.append('%s (score = %.2f%%)' % (human_string, score*100))
        # if human_string == obj and score > 0.8:
        #     frame_list.append(frame)
        #     frame_time.append(str(time*count))
        #     print (str(obj) + '가 나타난 시간 : ' + str(time * count) + '초')
    # 보드를 생성하고 결과를 출력합니다.
    board = white.copy()
    for i in range(len(text_list)):
        if i==0:
            cv2.putText(board, '%d.%s' %(i+1, text_list[i]), (OUT_X_ORIGIN, OUT_LINE_SPACE * i + OUT_Y_ORIGIN), cv2.FONT_HERSHEY_SIMPLEX, OUT_FONT_SIZE, (0, 0, 255))
        else:
            cv2.putText(board,'%d.%s' %(i+1, text_list[i]), (OUT_X_ORIGIN, OUT_LINE_SPACE * i + OUT_Y_ORIGIN), cv2.FONT_HERSHEY_SIMPLEX,
                    OUT_FONT_SIZE, (0, 0, 0))

    # 프레임 및 분석 결과를 출력합니다.
    cv2.imshow('webcam_detection', frame)
    cv2.imshow('board', board)

    # 프레임의 위치를 초기화합니다.
    if not moved:
        cv2.moveWindow('webcam_detection', FRAME_X_ORIGIN, FRAME_Y_ORIGIN)
        cv2.moveWindow('board', BOARD_X_ORIGIN, BOARD_Y_ORIGIN)
        moved = True

# 프로그램을 끝냅니다.
if video is not None:
    video.release()
cv2.destroyAllWindows()

# cnt = 0
# for frame in frame_list:
#     cv2.imshow('frame', frame)
#     print(obj + "나온 시간 : " + str(frame_time[cnt]))
#     cnt = cnt+1
#     if cv2.waitKey(1000) & 0xFF == ord('q'):
#         break