import numpy as np
# OpenCV 라이브러리를 불러옵니다.
import cv2
import tensorflow as tf, sys
import os
import time

# MP4 파일을 엽니다.
video = cv2.VideoCapture('testing.mp4')
grayscale = False

# 파일이 열려있는 동안 작업을 진행합니다.
# !!!!! IMPORTANT !!!!
# Python에서 None과의 비교를 위해 != 연산자를 사용하지 마십시오.

count = 1

while video is not None and video.isOpened():

    # 영상 파일로부터 데이터를 가져옵니다.
    ret, frame = video.read()
    if frame is None:
        break

    if grayscale is True:
        # 프레임을 흑백으로 변환합니다.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 흑백으로 변환된 프레임을 출력합니다.
        cv2.imshow('frame', gray)
    else:
        cv2.imshow('frame', frame)
        ch = cv2.waitKey(33)

        # Q를 누르면 종료합니다.
        if ch == 113:
            break;
	
	# S를 누르면 해당 프레임을 분석합니다.
        elif ch == 115:
            cv2.imwrite('frame%d.jpg' % count, frame)
            # #image_path = sys.argv[1]
            image_path = "frame" + str(count) + ".jpg"


            # Read in the image_data
            image_data = tf.gfile.FastGFile(image_path, 'rb').read()


            # Loads label file, strips off carriage return
            label_lines = [line.rstrip() for line
                           in tf.gfile.GFile("/home/chj/PycharmProjects/image_classify/retrained_labels.txt")]

            # Unpersists graph from file
            with tf.gfile.FastGFile("/home/chj/PycharmProjects/image_classify/retrained_graph.pb", 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(graph_def, name='')

            with tf.Session() as sess:
                # Feed the image_data as input to the graph and get first prediction
                softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

                predictions = sess.run(softmax_tensor, \
                                       {'DecodeJpeg/contents:0': image_data})


                # Sort to show labels of first prediction in order of confidence
                top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

                for node_id in top_k:
                    human_string = label_lines[node_id]
                    score = predictions[0][node_id]
                    print('%s (score = %.5f)' % (human_string, score))

            os.remove(image_path)

        count += 1

# 프로그램을 끝냅니다.
if video is not None:
    video.release()
cv2.destroyAllWindows()
