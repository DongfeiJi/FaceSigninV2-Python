# 离线版本：人脸验证-128维向量输出
import sys
import os
import dlib
import glob

# 重点！！！！：基于自己的路径进行调整
predictor_path = './model/shape_predictor_68_face_landmarks.dat'
faces_folder_path = './data/image/faces'
face_rec_model_path = './model/dlib_face_recognition_resnet_model_v1.dat'

# 模型导入
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

win = dlib.image_window()

# Now process all the images
for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)

    win.clear_overlay()
    win.set_image(img)

    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))

    # 逐个处理.
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # 特征点检测
        shape = sp(img, d)
        # 绘制特征点
        win.clear_overlay()
        win.add_overlay(d)
        win.add_overlay(shape)

        # 128维向量计算：输出特征向量
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        print(face_descriptor)

        # 通过欧式距离，计算相似度：
        dlib.hit_enter_to_continue()


