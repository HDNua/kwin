
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np


# In[2]:

# focus는 model 설계


# In[3]:

input_data = [[1, 5, 3, 7, 8, 10, 12],
              [5, 8, 10, 3, 9, 7, 1]]           
            

# 이것을 아래와 같이 인식시킨다.
# 즉, input_data가 들어가 training되어
# label_data가 출력되도록 한다.

label_data = [[0, 0, 0, 1, 0],
              [1, 0, 0, 0, 0]]

# input_data의 1행은 label_data의 1행과
# input_data의 2행은 label_data의 2행과 일치해야함


# In[4]:

# model을 설계하여 data를 넣는다.
# list로 되어있는 data를 tensorflow가 인식하도록 한다.


# In[5]:

INPUT_SIZE = 7
HIDDEN1_SIZE = 10
HIDDEN2_SIZE = 8
CLASSES = 5
# input의 size와 hidden layer의 수, output 결과의 class수를 설정해준다.

Learning_Rate = 0.05

x = tf.placeholder( dtype = tf.float32, shape = [None, INPUT_SIZE])
# shape의 배치사이즈는 None을 많이 쓴다.

y_ = tf.placeholder(dtype = tf.float32, shape = [None, CLASSES])
# y_로 선언해줌

# 또한 shape을 위와 같이 명시하는 것은 2차원 input data를 사용하는 것을 의미한다.


# In[6]:

tensor_map = {x : input_data, y_ : label_data}
# mapping
# session을 돌릴 때 x는 input_data로 y_는 label_data를 사용한다.


# In[7]:

# 신경망은 matrix연산이므로 weight를 설정해줘야한다.


# In[8]:

w_h1 = tf.Variable(tf.truncated_normal(shape = [INPUT_SIZE, HIDDEN1_SIZE]), dtype = tf.float32)
b_h1 = tf.Variable( tf.zeros(shape = [HIDDEN1_SIZE]), dtype = tf.float32)

w_h2 = tf.Variable(tf.truncated_normal(shape = [HIDDEN1_SIZE, HIDDEN2_SIZE]), dtype = tf.float32)
b_h2 = tf.Variable(tf.zeros(shape = [HIDDEN2_SIZE]), dtype = tf.float32)

w_o = tf.Variable(tf.truncated_normal(shape = [HIDDEN2_SIZE, CLASSES]), dtype = tf.float32)
b_o = tf.Variable(tf.zeros(shape = [CLASSES]), dtype = tf.float32)


# saver는 weight가 끝나는 시점에 생성하는게 좋다.
# 더 구체적으로 weight를 저장하고 싶다면 param_list 생성

param_list = {w_h1, b_h1, w_h2, b_h2, w_o, b_o}

saver = tf.train.Saver(param_list)
# saver를 저장하는 시점은 보통 print와 많이 수행한다.


# In[9]:

hidden1 = tf.sigmoid(tf.matmul(x, w_h1) + b_h1)
hidden2 = tf.sigmoid(tf.matmul(hidden1, w_h2) + b_h2)
y = tf.sigmoid(tf.matmul(hidden2, w_o) + b_o)


# In[10]:

# model 설계의 끝


# In[11]:

# tf.reduce_sum(input_tensor, reduction_indices=None, keep_dims=False, name=None)

# input_tensor은 matrix
# reduction_indices는 몇 개의 값을 보고싶은가?
# 어떻게 projection하여 보고싶은가?

# 'x' is [[1, 1, 1]
#         [1, 1, 1]]
# tf.reduce_sum(x) ==> 6 ; 하나의 값으로
# tf.reduce_sum(x, 0) ==> [2, 2, 2] ; column으로 projection
# tf.reduce_sum(x, 1) ==> [3, 3] ; row로 projection
# tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]] ; 차원 유지
# tf.reduce_sum(x, [0, 1]) ==> 6 ; 

# tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None)

# 평균을 내는 함수

# 'x' is [[1., 1.]
#         [2., 2.]]
# tf.reduce_mean(x) ==> 1.5
# tf.reduce_mean(x, 0) ==> [1.5, 1.5]
# tf.reduce_mean(x, 1) ==> [1.,  2.]


# In[12]:

# training을 위한 cost function을 정의해야한다.

cost = -y_ * tf.log(y) - (1-y_) * tf.log(1-y)
# 크로스엔트로피(cross_entropy)
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# tf.log(x, name=None)
# y = log_eX
# y_(label이 0일 때는 (1-y_) * tf.log(1-y) 를 사용

cost_sum = tf.reduce_sum(cost, reduction_indices = 1)

cost_mean = tf.reduce_mean(cost_sum)

# 목표는 mean을 최소화하는 방향으로

train = tf.train.GradientDescentOptimizer(Learning_Rate).minimize(cost_mean)

# tf.train.GradientDescentOptimizer(learning_rate).minimize() 이용


# In[13]:

sess = tf.Session()
# Session

init = tf.global_variables_initializer()
# 초기화

sess.run(init)

result_cost = sess.run(cost, feed_dict = tensor_map)
# mapping한 data 명시

print(result_cost)


# In[14]:

result_sum = sess.run(cost_sum, feed_dict = tensor_map)

print(result_sum)


# In[15]:

result_mean = sess.run(cost_mean, feed_dict = tensor_map)

print(result_mean)


# In[16]:

# tf.argmax(input, axis=None, name=None, dimension=None)
# input에서 가장 큰 index를 찾아준다.

# ex)

arg_input = [[0, 2, 4, 1, 7],
             [1, 6, 9, 5, 3]]
arg_y_ = tf.placeholder(dtype = tf.float32, shape = [None, 5])

ex_argmax = tf.argmax(arg_y_, 1)

print (sess.run(ex_argmax, feed_dict={arg_y_:arg_input}))


# In[17]:

# 그러므로 accuracy를 계산하는 방법은
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# equal은 두 인자가 같은지 다른지 확인하여 bool typl으로 return
# correct_prediction의 결과로 [True, False, True, True]가 나왔다면
# [1,0,1,1]로 볼 수 있고, 이것을 reduce_mean을 해준다.
# accuracy :0.75
# 여기서 y는 training된 결과, y_는 label_data
# 만족할만한 결과가 나올때까지 training시키는 것 !

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype = tf.float32))


# In[19]:

for i in range(1001):
    _, result_train, accuracy_ = sess.run([train, cost_mean, accuracy], feed_dict = tensor_map)
    if i % 100 == 0 :
        saver.save(sess, './tensorflow.live.checkpoint')
        print ("Step :", i)
        print ("loss :", result_train)
        print ("accruracy : ", accuracy_, "\n")


# In[24]:

sess.close()

