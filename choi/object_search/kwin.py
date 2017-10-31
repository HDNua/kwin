import os
import getpass
from datetime import datetime


# kwin 프로젝트에서 사용할 경로 명을 초기화해주면 편하다.
_, PROJECT_NAME = os.path.split(os.getcwd())  # "image_rotator"
RESOURCE_NAME = "object_search"
DATASET_NAME = "data"

#
user_name = "lab"
user_os = "win64"

#
if user_name == "handy":
    if user_os == "win64":
        # 한 도영의 Windows 64bit 설정
        KWIN_DIR = "D:/handy/work/project/kwin/kwin4"
        PROJECT_DIR = "%s/handy/%s" %(KWIN_DIR, PROJECT_NAME)
        RESOURCE_DIR = "D:/handy/work/project/kwin/resources/%s" %(RESOURCE_NAME)
    else:
        # 한 도영의 Linux 64bit 설정
        KWIN_DIR = "/home/handy/kwin" # kwin에 대한 심볼릭 링크를 홈 폴더에 생성하십시오.
        PROJECT_DIR = "%s/handy/%s" %(KWIN_DIR, PROJECT_NAME)
        RESOURCE_DIR = "%s/resources/%s" %(KWIN_DIR, RESOURCE_NAME)

elif user_name == "lab": # 변수 정의 예제
    KWIN_DIR = "C:/Users/idea_kwin/Desktop/work/kwin/kwin"
    PROJECT_DIR = "%s/handy/%s" %(KWIN_DIR, PROJECT_NAME)
    RESOURCE_DIR = "C:/Users/idea_kwin/Desktop/work/kwin/resources/%s" %(RESOURCE_NAME)

elif False: # 변수 정의 예제
    KWIN_DIR = "<kwin 폴더 경로>/kwin"
    PROJECT_DIR = "<프로젝트 폴더 경로>"
    RESOURCE_DIR = "<리소스 폴더 경로>"

else:
    print("KWIN_DIR, PROJECT_DIR, RESOURCE_DIR 변수를 정의하십시오.")
    print("KWIN_DIR: kwin 폴더의 경로입니다.")
    print("PROJECT_DIR: 작업할 프로젝트의 경로입니다.")
    print("RESOURCE_DIR: 프로젝트가 사용할 리소스가 담긴 폴더입니다.")
    raise Exception()


def kwin_path(name=""):
    """
    kwin 디렉터리로부터 주어진 이름을 갖는 파일의 전체 경로를 획득합니다.

    :param name: 추가로 전달할 이름입니다. 없으면 KWIN_DIR을 반환합니다.

    :return: kwin 디렉터리로부터 주어진 이름을 갖는 파일의 전체 경로를 획득합니다.
    """
    if name == "":
        return KWIN_DIR
    return KWIN_DIR + '/' + name


def project_path(name=""):
    """
    프로젝트 경로로부터 주어진 이름을 갖는 파일의 전체 경로를 획득합니다.

    :param name: 추가로 전달할 이름입니다. 없으면 PROJECT_DIR을 반환합니다.

    :return: 프로젝트 경로로부터 주어진 이름을 갖는 파일의 전체 경로를 획득합니다.
    """
    if name == "":
        return PROJECT_DIR
    return PROJECT_DIR + '/' + name


def resource_path(name=""):
    """
    리소스 경로로부터 주어진 이름을 갖는 파일의 전체 경로를 획득합니다.

    :param name: 추가로 전달할 이름입니다. 없으면 RESOURCE_DIR을 반환합니다.

    :return: 리소스 경로로부터 주어진 이름을 갖는 파일의 전체 경로를 획득합니다.
    """
    if name == "":
        return RESOURCE_DIR
    return RESOURCE_DIR + '/' + name


#
def print_now():
    print(datetime.now())





# 생성한 프로젝트 정보를 출력합니다.
if __name__ != "__main__":
    print("kwin path is [%s]" %(KWIN_DIR))
    print("project name is [%s]" %(PROJECT_NAME))
    print("project path is [%s]" %(PROJECT_DIR))
    print("resource path is [%s]" %(RESOURCE_DIR))
