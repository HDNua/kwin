"""

"""
from kwin import *




test_name = "000/000004.jpg"

test_name = "180/000023.jpg"


#
def train_path():
    return resource_path("train/%s" %(DATASET_NAME))


#
def test_path(name=""):
    if name == "":
        return resource_path("test/%s" %(DATASET_NAME))
    return resource_path("test/%s/%s" %(DATASET_NAME, name))