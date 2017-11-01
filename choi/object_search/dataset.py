"""
 dataset

 Developer: HeeJun Choi, DoYeong Han
 Version: 0.1.0
 Release Date: 2017-09-30
"""
from kwin import *

test_name = "000/000004.jpg"
test_name = "180/000023.jpg"


def train_path(name=""):
    """

    :param name:
    :return:
    """
    if name == "":
        return resource_path("train")
    return resource_path("train/%s" %name)


def train_data_path(name=""):
    """

    :param name:
    :return:
    """
    if name == "":
        return resource_path("train/%s" %DATASET_NAME)
    return resource_path("train/%s/%s" %(DATASET_NAME, name))


def test_path(name=""):
    """

    :param name:
    :return:
    """
    if name == "":
        return resource_path("test")
    return resource_path("test/%s" %name)


def test_data_path(name=""):
    """

    :param name:
    :return:
    """
    if name == "":
        return resource_path("test/%s" %DATASET_NAME)
    return resource_path("test/%s/%s" %(DATASET_NAME, name))


#
TRAIN_DIR = "C:/Users/idea_kwin/Desktop/work/kwin/train/%s" %(RESOURCE_NAME)


def bottleneck_dir():
    return "%s/bottlenecks" %TRAIN_DIR


def model_dir():
    return "%s/inception" %TRAIN_DIR


def output_graph():
    return "%s/train_graph.pb" %TRAIN_DIR


def output_labels():
    return "%s/train_labels.txt" %TRAIN_DIR


"""
BOTTLENECK_DIR = "C:/Users/idea_kwin/Desktop/work/kwin/train"
MODEL_DIR = "C:/Users/idea_kwin/Desktop/work/kwin/train"
OUTPUT_GRAPH = "C:/Users/idea_kwin/Desktop/work/kwin/train"
OUTPUT_LABELS = "C:/Users/idea_kwin/Desktop/work/kwin/train"
"""
