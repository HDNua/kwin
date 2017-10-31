import tensorflow as tf, sys
import cv2
from kwin import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#image_path = "test/cat.jpg"

image_list = ['test/frodo_1.jpg', 'test/frodo_2.jpg', 'test/frodo_3.jpg', 'test/frodo_4.jpg']

count = 1

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
               in tf.gfile.GFile("train/train_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("train/train_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

for image_path in image_list:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(image_path + str(count))
    count = count+1

    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()



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
        print(' ')