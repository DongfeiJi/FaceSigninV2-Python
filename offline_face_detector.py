# 离线版本：基于5特征点的人脸检测

import sys
import dlib
import os
import glob
import cv2

detector = dlib.get_frontal_face_detector()
win = dlib.image_window()
# 重要！！！：待进行人脸检测的图像文件夹地址，执行时需要根据自己的情况进行设置
faces_folder_path = './data/image/faces'
# sys.argv[1:]
for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("处理文件格式：: {}".format(f))
    img = dlib.load_rgb_image(f)
    dets = detector(img, 1)
    print("检测到人脸数量: {}".format(len(dets)))
    for i, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, d.left(), d.top(), d.right(), d.bottom()))

    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(dets)
    dlib.hit_enter_to_continue()
