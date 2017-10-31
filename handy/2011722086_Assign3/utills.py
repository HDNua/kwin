# -*- coding: utf-8 -*-
"""
Created on Tue May 30 16:42:28 2017

@author: Minsooyeo
"""
import os
import tensorflow as tf
import numpy as np


def search(dirname):
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        return filenames
        

#######################################################################################
# padding 전역 변수입니다. 0이면 'VALID', 1이면 'SAME'입니다.
padding = 1


def _CNNModel(img_width, img_height, kernel_info):
    """
    CNN 모델을 생성하여 반환합니다.

    :param img_width: 입력 이미지의 가로 길이입니다.
    :param img_height: 입력 이미지의 세로 길이입니다.
    :param kernel_info: 각 커널의 정보[커널 너비, 커널 높이, 커널 출력 수, 풀링 여부]입니다.
    :return: CNN 풀링층의 결과를 반환합니다.
    """
    global padding
    x = tf.placeholder(tf.float32, [None, img_width * img_height]) # (, dimension) 차원의 벡터를 생성합니다.

    # 제 1 콘볼루션 레이어입니다.
    k_width = kernel_info[0][0]
    k_height = kernel_info[0][1]
    k_outlayer_count = kernel_info[0][2]
    k_pool = kernel_info[0][3]

    x_img = tf.reshape(x, [-1, img_width, img_height, 1])
    W_conv_beg = weight_variable([k_width, k_height, 1, k_outlayer_count])
    b_conv_beg = bias_variable([k_outlayer_count])
    h_conv_beg = tf.nn.relu(Conv2d(x_img, W_conv_beg, padding=padding) + b_conv_beg)
    if k_pool is True:
        h_pool_beg = max_pool_2x2(h_conv_beg, padding=padding)
        # h_pool_beg = tf.nn.dropout(h_pool_beg, keep_prob=keep_prob)
    else:
        h_pool_beg = h_conv_beg

    W_conv_list = [W_conv_beg]
    b_conv_list = [b_conv_beg]
    h_conv_list = [h_conv_beg]
    h_pool_list = [h_pool_beg]
    k_outlayer_count_prev = k_outlayer_count
    h_pool_prev = h_pool_beg
    h_pool = h_pool_beg
    for i in range(1, len(kernel_info)):
        k_width = kernel_info[i][0]
        k_height = kernel_info[i][1]
        k_outlayer_count = kernel_info[i][2]
        k_pool = kernel_info[i][3]

        W_conv = weight_variable([k_width, k_height, k_outlayer_count_prev, k_outlayer_count])
        b_conv = bias_variable([k_outlayer_count])
        h_conv = tf.nn.relu(Conv2d(h_pool_prev, W_conv, padding=padding) + b_conv)
        if k_pool is True:
            h_pool = max_pool_2x2(h_conv, padding=padding)
            # h_pool = tf.nn.dropout(h_pool, keep_prob=keep_prob)
        else:
            h_pool = h_conv

        k_outlayer_count_prev = k_outlayer_count
        h_pool_prev = h_pool

        W_conv_list.append(W_conv)
        b_conv_list.append(b_conv)
        h_conv_list.append(h_conv)
        h_pool_list.append(h_pool)

    # 반환할 모델은 마지막 풀링층의 결과입니다.
    model = h_pool
    print("%s ---> CNN Model was built" %(model))
    return model, x


def _FlatModel(h_pool2, fc_outlayer_count=256):
    """
    모델을 평평하게 펴줍니다.

    :param h_pool2: 평평하게 만들 모델입니다.
    :return: 평평해진 모델을 반환합니다.
    """
    global padding
    h_pool2_shape = h_pool2.get_shape().as_list()
    flat_size = h_pool2_shape[1] * h_pool2_shape[2] * h_pool2_shape[3]

    W_fc1 = weight_variable([flat_size, fc_outlayer_count])
    b_fc1 = bias_variable([fc_outlayer_count])
    h_pool2_flat = tf.reshape(h_pool2, [-1, flat_size])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # 반환할 모델은 평평해진 모델입니다.
    model = h_fc1
    print("%s ---> Flat Model was built" %(model))
    return model


def _DropOut(h_fc1):
    """
    Overfitting을 줄이기 위해 dropout을 수행합니다.

    :param h_fc1: Dropout을 수행할 입력입니다.
    :return: Dropout이 수행된 모델을 반환합니다.
    """
    keep_prob = tf.placeholder(tf.float32)
    dropout_model = tf.nn.dropout(h_fc1, keep_prob)
    print("%s ---> DropOut Model was built" %(dropout_model))
    return dropout_model, keep_prob


def _SoftMax(h_fc1_drop, label_count, fc_outlayer_count=256):
    """
    Softmax 함수를 적용합니다.

    :param h_fc1_drop: Softmax를 적용할 입력입니다.
    :return: Softmax가 적용된 모델을 반환합니다.
    """
    # Readout Layer
    W_fc2 = weight_variable([fc_outlayer_count, label_count])
    b_fc2 = bias_variable([label_count])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    #
    model = y_conv
    print("%s ---> SoftMax Model was built" %(model))
    return model


def _SetAccuracy(y_conv, label_count): # (h_fc1_drop, value)
    """
    Accuracy를 계산합니다.

    :param y_conv: Accuracy를 계산할 모델입니다.
    :param label_count: 레이블 수입니다.
    :return: (train_step, accuracy, y_, correct_prediction) 튜플을 반환합니다.
    """
    train_step = None
    accuracy = None
    y_ = tf.placeholder(tf.float32, [None, label_count])
    correct_prediction = None

    #
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #
    print("%s ---> Train Model was built" %(train_step))
    print("%s ---> Accuracy Model was built" %(accuracy))
    return train_step, accuracy, y_, correct_prediction


def Nextbatch(data, label, batch_size):
    """
    다음 배치를 생성합니다.

    :param data: 훈련 집합의 데이터입니다.
    :param label: 훈련 집합의 레이블입니다.
    :param batch_size: 배치 크기입니다.
    :return: 임의의 인덱스에 대한 데이터와 레이블 리스트를 (data, label) 형태의 튜플로 반환합니다.
    """
    indices = np.random.permutation(data.shape[0])[:batch_size]
    ret_data = data[indices]
    ret_label = label[indices]
    return ret_data, ret_label


def weight_variable(shape):
    """
    가중치 변수를 생성합니다.

    :param shape: 가중치 변수의 형태입니다.
    :return: shape 모양의 표준편차 0.1인 Truncated Normal Distribution으로부터 생성된 난수로 가득 찬
             TensorFlow 변수를 생성하여 반환합니다.
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """
    편향 변수를 생성합니다.

    :param shape: 편향 변수의 형태입니다.
    :return: shape 모양의 0.1로 가득 찬 TensorFlow 변수를 생성하여 반환합니다.
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def Conv2d(x, w, padding):
    """
    w 필터를 이용하여 x와 콘볼루션을 수행합니다.

    :param x: 입력입니다.
    :param w: 필터입니다.
    :param padding: 0이면 'VALID', 1이면 'SAME'입니다.
    :return: x와 w의 콘볼루션 결과를 반환합니다.
    """
    _padding = None
    if padding == 0:
        _padding = 'VALID'
    elif padding == 1:
        _padding = 'SAME'
    else:
        raise Exception("padding is neither 0 nor 1.")
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding=_padding)


def max_pool_2x2(x, padding):
    """
    2x2 크기의 Max Pooling을 수행합니다.

    :param x: 입력 벡터입니다.
    :param padding: 0이면 'VALID', 1이면 'SAME'입니다.
    :return: 2x2 크기의 Max Pooling 결과를 반환합니다.
    """
    _padding = None
    if padding == 0:
        _padding = 'VALID'
    elif padding == 1:
        _padding = 'SAME'
    else:
        raise Exception("padding is neither 0 nor 1.")
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=_padding)
