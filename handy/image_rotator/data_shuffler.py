"""
data_shuffler.py

Developer: DoYoung Han
Version: 4.0.0
Description: 데이터 세트를 섞습니다.
"""
import sys
import os
import random
from kwin import *
import dataset


#
def shuffle(target_dir):
    """
    데이터 세트를 섞습니다.

    :param target_dir: 데이터 세트가 담긴 폴더입니다.
    """
    # 이름 충돌을 방지하기 위해 임시로 이름을 변경합니다.
    src_names = os.listdir(target_dir)
    for filename in src_names:
        os.rename(target_dir + filename, target_dir + "_" + filename)

    # 파일의 순서를 섞습니다.
    src_names = os.listdir(target_dir)
    random.shuffle(src_names)
    i = 1

    # 섞인 순서대로 이름을 다시 부여합니다.
    for filename in src_names:
        if i % 100 == 0:
            print("%dth work has been completed" % (i))

        _, file_extension = os.path.splitext(filename)
        new_name = target_dir + str("%06i" %(i)) + file_extension
        os.rename(target_dir + filename, new_name)
        i += 1

    print("All works of [%s] are done" % (target_dir))


def reset_name(target_dir):
    """
    데이터 세트의 이름을 초기화합니다.

    :param target_dir: 데이터 세트가 담긴 폴더입니다.
    """
    # 이름 충돌을 방지하기 위해 임시로 이름을 변경합니다.
    src_names = os.listdir(target_dir)
    for filename in src_names:
        os.rename(target_dir + filename, target_dir + "_" + filename)

    # 이름을 다시 부여합니다.
    i = 1
    src_names = os.listdir(target_dir)
    for filename in src_names:
        os.rename(target_dir + filename, target_dir + "%06d.jpg" %(i))
        i += 1


def add_original_name(target_dir, dirname):
    """
    데이터 세트의 맨 앞에 경로의 이름을 추가합니다.

    :param target_dir: 데이터 세트가 담긴 폴더입니다.
    :param dirname: 경로의 이름입니다.
    """
    src_names = os.listdir(target_dir)
    for filename in src_names:
        os.rename(target_dir + filename, target_dir + dirname + filename)


#
cwd = dataset.test_path() + '/'  # resource_path("wr") + '/'  # sys.argv[1]

cwd = dataset.train_path() + '/'

print('current working directory is [%s]' %(cwd))

#
src_names = os.listdir(cwd)
if True:
    for filename in src_names:
        target_dir = cwd + filename + '/'
        shuffle(target_dir)
elif True:
    for filename in src_names:
        target_dir = cwd + filename + '/'
        reset_name(target_dir)
else:
    for filename in src_names:
        target_dir = cwd + filename + '/'
        add_original_name(target_dir, filename)

#
print("All works are done")
