# -*- coding: utf-8 -*-
"""
Created on Tue May 30 16:43:10 2017
☜☜☜☜☜☜★☆★☆★☆★☆ provided code ★☆★☆★☆★☆☞☞☞☞☞☞
@author: Minsooyeo
"""

import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image as im
import numpy as np
import utills as ut
import tensorflow as tf
sess = tf.InteractiveSession()
train_epoch = 5000


#
FLAG_FINGER = 0
FLAG_FACE = 1
FLAG_ANGLE = 2
flag = FLAG_ANGLE

#
if flag is FLAG_FINGER:
    class_num = 5
    additional_path = '\\finger\\'
elif flag is FLAG_FACE:
    class_num = 6
    additional_path = '\\face\\'
elif flag is FLAG_ANGLE:
    class_num = 4
    additional_path = '\\angle\\'
else:
    raise Exception("Unknown flag %d" %flag)

# define parameter
data_length = []
dir_image = []
data = []
label = []
data_shape = [298, 298]
current_pwd = os.getcwd()
for i in range(class_num):
    dir_image.append(ut.search(current_pwd + additional_path + str(i + 1)))
    data_length.append(len(dir_image[i]))
    data.append(np.zeros([data_length[i], data_shape[1], data_shape[0]]))
    label.append(np.zeros([data_length[i], class_num]))
    label[i][:, i] = 1

# load data
for q in range(class_num):
    for i in range(data_length[q]):
        if i % 100 == 0:
            print("%dth data is opening" %i)

        data[q][i, :, :] = np.mean(im.open(current_pwd + additional_path + str(q + 1) + '\\' + dir_image[q][i]), -1)


if flag is FLAG_FINGER:
    rawdata = np.concatenate((data[0], data[1], data[2], data[3], data[4]), axis=0)
    raw_label = np.concatenate((label[0], label[1], label[2], label[3], label[4]), axis=0)
elif flag is FLAG_FACE:
    rawdata = np.concatenate((data[0], data[1], data[2], data[3], data[4], data[5]), axis=0)
    raw_label = np.concatenate((label[0], label[1], label[2], label[3], label[4], label[5]), axis=0)
elif flag is FLAG_ANGLE:
    rawdata = np.concatenate((data[0], data[1], data[2], data[3]), axis=0)
    raw_label = np.concatenate((label[0], label[1], label[2], label[3]), axis=0)
else:
    raise Exception("Unknown class number %d" %class_num)

del data
del label

total_data_poin = rawdata.shape[0]
permutation = np.random.permutation(total_data_poin)
rawdata = rawdata[permutation, :, :]
raw_label = raw_label[permutation, :]

rawdata = np.reshape(rawdata, [rawdata.shape[0], data_shape[0] * data_shape[1]])

########################################################################################################
#
img_width = data_shape[0]
img_height = data_shape[1]

if flag is FLAG_FINGER:
    train_count = 5000 # 손가락 인식을 테스트하려는 경우 이 부분을 수정하십시오. (2000 또는 5000으로 테스트함)
    test_count = 490
elif flag is FLAG_FACE:
    train_count = 2000  # train data 수가 5000개가 안 돼서 또는 overfitting에 의해 NaN 문제가 발생합니다. 값을 바꾸지 마십시오!
    test_count = 490
elif flag is FLAG_ANGLE:
    train_count = 6000  # train data 수가 5000개가 안 돼서 또는 overfitting에 의해 NaN 문제가 발생합니다. 값을 바꾸지 마십시오!
    test_count = 1000
else:
    raise Exception("unknown flag %d" %flag)

#
train_epoch = train_count


#
TrainX = rawdata[:train_count] # mnist.train.images
TrainY = raw_label[:train_count] # mnist.train.labels
testX = rawdata[train_count:train_count+test_count] # mnist.test.images
testY = raw_label[train_count:train_count+test_count] # mnist.test.labels

# 손가락 구분을 테스트하기 위해 층의 수를 바꾸는 경우 else 부분을 수정하십시오.
if flag is FLAG_FINGER: # 손가락 구분의 경우 층에 따라 경우를 테스트하려면 이 부분을 수정하십시오.
    CNNModel, x = ut._CNNModel(img_width=img_width, img_height=img_height,
                               kernel_info=[
                                   [3, 2, 32, True],
                                   [3, 2, 64, True],
                                   [3, 2, 128, True],
                                   [3, 2, 64, True],
                                   [3, 2, 128, True],
                                   # [3, 2, 128, True],
                               ])
elif flag is FLAG_FACE: # 얼굴 인식의 경우 2개의 층만으로도 구분이 완전히 잘 됩니다. 층의 수를 수정하지 마십시오.
    CNNModel, x = ut._CNNModel(img_width=img_width, img_height=img_height,
                               kernel_info=[
                                   [3, 2, 32, True],
                                   [3, 2, 64, True],
                                   # [3, 2, 128, True],
                                   # [3, 2, 64, True],
                                   # [3, 2, 128, True],
                                   # [3, 2, 128, True],
                               ])
elif flag is FLAG_ANGLE: #
    CNNModel, x = ut._CNNModel(img_width=img_width, img_height=img_height,
                               kernel_info=[
                                   [1, 1, 32, True],
                                   # [1, 1, 64, True],
                                   # [1, 1, 128, True],
                                   # [1, 1, 64, True],
                                   # [1, 1, 128, True],
                                   # [3, 2, 128, True],
                               ])
else:
    raise Exception("Unknown flag %d" %flag)

FlatModel = ut._FlatModel(CNNModel, fc_outlayer_count=128)
DropOut, keep_prob = ut._DropOut(FlatModel)
SoftMaxModel = ut._SoftMax(DropOut, label_count=class_num, fc_outlayer_count=128)
TrainStep, Accuracy, y_, correct_prediction = ut._SetAccuracy(SoftMaxModel, label_count=class_num)

sess.run(tf.global_variables_initializer())

for i in range(train_epoch):
    tmp_trainX, tmp_trainY = ut.Nextbatch(TrainX, TrainY, 50)
    if i%100 == 0:
        train_accuracy = Accuracy.eval(feed_dict={x: tmp_trainX, y_: tmp_trainY, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    TrainStep.run(feed_dict={x: tmp_trainX, y_: tmp_trainY, keep_prob: 0.7})

print("test accuracy %g" %Accuracy.eval(feed_dict={x: testX[1:1000, :], y_: testY[1:1000], keep_prob: 1.0}))