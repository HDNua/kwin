"""
 webcam

 Developer: HeeJun Choi, DoYoung Han
 Version: 1.0.0
 Release Date: 2017-09-30
"""
import cv2


def record_avi(target_name, target_dir):
    """

    :param target_name:
    :param target_dir:
    :return:
    """
    cap = cv2.VideoCapture(0)
    frame_list = []

    ########################################################################
    # 파일이 열려있는 동안 작업을 진행합니다.
    # !!!!! IMPORTANT !!!!
    # Python에서 None과의 비교를 위해 != 연산자를 사용하지 마십시오.
    count = 0
    while cap is not None and cap.isOpened():
        # 영상 파일로부터 데이터를 가져옵니다.
        ret, frame = cap.read()
        count += 1
        if frame is None:
            break
        elif ret is True:
            cv2.imshow('frame', frame)
            frame_list.append(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # 프로그램을 끝냅니다.
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

    #
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out = cv2.VideoWriter(filename="%s/%s.avi" %(target_dir, target_name),
                          fourcc=fourcc,
                          fps=20.0,
                          frameSize=(640, 480),
                          isColor=True)

    frame_list_count = len(frame_list)
    for i in range(frame_list_count):
        frame = frame_list[i]
        percentage = i / frame_list_count * 100
        if percentage % 10 == 0:
            print("%3d%% complete" % percentage)
        out.write(frame)

        cv2.imwrite('%s/%010d.jpg' %(target_dir, i), frame)

    if out is not None:
        out.release()
