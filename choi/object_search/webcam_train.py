"""
 webcam_train

 Developer: HeeJun Choi, DoYoung Han
 Version: 0.1.0
 Release Date: 2017-09-30
"""

import numpy as np
import cv2
import tensorflow as tf
import sys
from kwin import *
import time
import webcam
import dataset

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

########################################################################
#
########################################################################
#
while True:
    target_name = input("무엇을 학습할까요? (종료하려면 exit) ")

    target_name=target_name.strip()

    target_dir = "%s/%s" %(dataset.train_data_path(), target_name)
    if target_name == 'exit':
        break

    if os.path.exists(target_dir) is False:
        os.mkdir(target_dir)
    webcam.record_avi(target_name=target_name, target_dir=target_dir)
    print("[%s]에 대한 동영상 촬영이 완료되었습니다." %target_name)

#
print("학습을 시작합니다. 종료 메시지가 나타날 때까지 잠시 기다리십시오.")

#
import retrain
retrain.do_train()

#
print("학습이 종료되었습니다. bottleneck을 확인하십시오.")

