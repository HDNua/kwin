import os
import getpass

# kwin 디렉터리의 경로입니다.
USER_NAME = getpass.getuser()
KWIN_DIR = "/home/" + USER_NAME + "/kwin/"  # kwin에 대한 심볼릭 링크를 홈 폴더에 생성하십시오.
OWNER_DIR = "choi"

# 프로젝트 정보입니다.
_, PROJECT_NAME = os.path.split(os.getcwd())  # "image_classify"
PROJECT_DIR = KWIN_DIR + OWNER_DIR + "/" + PROJECT_NAME + "/"  # 모든 디렉터리의 끝에 '/' 기호를 추가하십시오.

# 리소스 경로입니다.
RESOURCE_DIR = KWIN_DIR + "resources/" + PROJECT_NAME + "/"


def kwin_path(name):
    """
    kwin 디렉터리로부터 주어진 이름을 갖는 파일의 전체 경로를 획득합니다.

    :param name: 추가로 전달할 이름입니다. 없으면 KWIN_DIR을 반환합니다.

    :return: kwin 디렉터리로부터 주어진 이름을 갖는 파일의 전체 경로를 획득합니다.
    """
    return KWIN_DIR + name


def project_path(name):
    """
    프로젝트 경로로부터 주어진 이름을 갖는 파일의 전체 경로를 획득합니다.

    :param name: 추가로 전달할 이름입니다. 없으면 PROJECT_DIR을 반환합니다.

    :return: 프로젝트 경로로부터 주어진 이름을 갖는 파일의 전체 경로를 획득합니다.
    """
    return PROJECT_DIR + name


def resource_path(name):
    """
    리소스 경로로부터 주어진 이름을 갖는 파일의 전체 경로를 획득합니다.

    :param name: 추가로 전달할 이름입니다. 없으면 RESOURCE_DIR을 반환합니다.

    :return: 리소스 경로로부터 주어진 이름을 갖는 파일의 전체 경로를 획득합니다.
    """
    return RESOURCE_DIR + name


# 생성한 프로젝트 정보를 출력합니다.
if __name__ != "__main__":
    print(KWIN_DIR)
    print(PROJECT_NAME)
    print(PROJECT_DIR)
    print(RESOURCE_DIR)
