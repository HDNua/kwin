"""
 video recognizer

 Developer: HeeJun Choi, DoYeong Han
 Version: 0.5.0
 Release Date: 2017-01-09
"""

import numpy as np
import cv2
import tensorflow as tf
import sys
import os
from kwin import *

########################################################################
# 프로그램을 초기화합니다.
########################################################################
# MP4 파일을 엽니다.
video = cv2.VideoCapture(0)

# 파일을 열 수 없는 경우 프로그램을 종료합니다.
if video is None or video.isOpened() == False:
    print("cannot open webcam")
    sys.exit()

# 영상의 프레임을 그레이스케일로 가져오려면 True로 설정합니다.
grayscale = False

# 레이블 파일을 가져오고(GFile) 모든 줄의 끝에 있는 캐리지 리턴을 제거합니다(rstrip).
label_lines = [line.rstrip() for line
               in tf.gfile.GFile(project_path("retrained_labels.txt"))]

# 그래프 파일로부터 그래프를 생성합니다.
with tf.gfile.FastGFile(project_path("retrained_graph.pb"), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

# 세션을 생성합니다.
with tf.Session() as sess:
    # image_data를 그래프에 입력으로 주고 1차 예측(First prediction)을 획득합니다.
    #  (Feed the image_data as input to the graph and get first prediction)
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

########################################################################
# 파일이 열려있는 동안 작업을 진행합니다.
# !!!!! IMPORTANT !!!!
# Python에서 None과의 비교를 위해 != 연산자를 사용하지 마십시오.
text = ''
while True: # and video.isOpened():
    # 영상 파일로부터 데이터를 가져옵니다.
    ret, frame = video.read()
    if frame is None:
        break

    # 화면에 출력할 텍스트의 리스트를 정의합니다.
    text_list = []

    # 그레이스케일 플래그가 참입니다.
    if grayscale:
        # 프레임을 흑백으로 변환합니다.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 흑백으로 변환된 프레임을 출력합니다.
        cv2.imshow('frame', gray)

    # 그레이스케일이 아닙니다.
    else:
        # 이미지를 화면에 표시합니다.
        # cv2.imshow('frame', frame)

        # cv2.waitKey(int delay) -> https://goo.gl/lhjVHH
        # delay 밀리초마다 키 이벤트를 대기합니다.
        ch = cv2.waitKey(1)

        # Q를 누르면 종료합니다.
        if ch == 113:
            break
        # S를 누르면 해당 프레임을 분석합니다.
        elif ch == 115:
            # Numpy 행렬 형태로 프레임을 변환합니다.
            #  0:3 -> RGB 값을 유지합니다.
            pil_img = np.array(frame)[:, :, 0:3]

            # 1차 예측의 확률을 획득합니다.
            predictions = sess.run(softmax_tensor, {'DecodeJpeg:0': pil_img})

            print("############################################################")

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
                print('%s (score = %.5f)' % (human_string, score))

                # 이미지를 화면에 표시합니다.
                text_list.append('%s (score = %.5f)' % (human_string, score))

            # 분석할 이미지를 화면에 출력합니다. 출력하지 않으려면 문장을 주석 처리하십시오.
            cv2.imshow('분석할 이미지', pil_img)
        else:
            # Numpy 행렬 형태로 프레임을 변환합니다.
            #  0:3 -> RGB 값을 유지합니다.
            pil_img = np.array(frame)[:, :, 0:3]

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

                # 이미지를 화면에 표시합니다.
                text_list.append('%s (score = %.5f)' % (human_string, score))

        #
        xOffset = 50
        yOffset = 50
        yDist = 50
        for i in range(len(text_list)):
            cv2.putText(frame, text_list[i], (xOffset, yDist * i + yOffset),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        cv2.imshow('frame', frame)

# 프로그램을 끝냅니다.
if video is not None:
    video.release()
cv2.destroyAllWindows()