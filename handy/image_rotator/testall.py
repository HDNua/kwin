



import tensorflow as tf
from kwin import *

from PIL import Image
import dataset


#
parent_dir = TRAIN_DIR + '/'

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line in tf.gfile.GFile(parent_dir + "retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile(parent_dir + "retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


#
# 맞으면 그대로 둔다.
# 틀리면 src-dst 폴더로 옮긴다.
# (ex) 000 폴더에 있는 사진이 270으로 나타난다면
# 000-270 폴더를 만들고 그 폴더로 옮긴다.
# 이는 사진 자체가 좋은 타겟이 아닐 가능성이 있기 때문.
#
# 조건문을 함수 형태로 만들어서 조건을 수정하게 쉽게 구현하자.
#
def testall_target(target_dir):
    dirnames = os.listdir(target_dir)
    dirnames.reverse()

    for dirname in dirnames:
        testdir_path = "%s/%s" % (target_dir, dirname)
        test_target(origin_angle=dirname, target_dir=testdir_path, base_dir=target_dir)
        print("All works of [%s] are done" % (dirname))

        # if dirname == "000" or dirname == "090" or dirname == "180" or dirname == "270":


def test_target(origin_angle, target_dir, base_dir):
    filenames = os.listdir(target_dir)

    i = 1
    for filename in filenames:
        if i % 10 == 0:
            print("%dth done" %i)

        # Read in the image_data
        ## image_path = dataset.test_path(filepath)  # (sys.argv[1])
        image_path = "%s/%s" %(target_dir, filename)
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()

        classify(origin_angle=origin_angle, image_name=filename, image_path=image_path, image_data=image_data, base_dir=base_dir)
        i += 1


def classify(origin_angle, image_name, image_path, image_data, base_dir):
    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        for node_id in top_k:
            target_angle = label_lines[node_id]
            score = predictions[0][node_id]

            if False:
                ## print('%s (score = %.5f)' % (human_string, score))
                # 레이블이 맞는 경우 그대로 둡니다.
                if target_angle == origin_angle:
                    pass
                else:
                    newpath_dir = "%s/%s-%s" % (base_dir, origin_angle, target_angle)
                    if os.path.exists(newpath_dir) is False:
                        os.mkdir(newpath_dir)

                    newpath = "%s/%s-%s" %(newpath_dir, origin_angle, image_name)
                    print("Weird data moved from [%s] to [%s]" %(image_path, newpath))

                    if False:
                        os.rename(image_path, newpath)
                    else:
                        rotate_image(image_path=image_path, origin_angle=origin_angle, target_angle=target_angle, dst_path=newpath)
            else:
                to_angle = "000"
                newpath_dir = "%s/%s-%s" % (base_dir, target_angle, to_angle)
                if os.path.exists(newpath_dir) is False:
                    os.mkdir(newpath_dir)

                newpath = "%s/[%s-%s]-%s" % (newpath_dir, origin_angle, target_angle, image_name)
                rotate_image(image_path=image_path, origin_angle=to_angle, target_angle=target_angle, dst_path=newpath)

            break


def rotate_image(image_path, origin_angle, target_angle, dst_path):
    img = Image.open(image_path)
    angle_dif = (360 + int(target_angle) - int(origin_angle)) % 360
    img_out = img.rotate(angle_dif, expand=True)
    img_out.save(dst_path)


#
target_dir = dataset.test_path()
testall_target(target_dir)
