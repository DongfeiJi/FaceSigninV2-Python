# 离线版本：人脸对齐
import sys
import dlib

# 重要！！！：运行时，基于自己的路径做修改
predictor_path = './model/shape_predictor_5_face_landmarks.dat'  # 5点模型文件
face_file_path = './data/image/faces/bald_guys.jpg' # 人脸照片路径

# 人脸检测器
detector = dlib.get_frontal_face_detector()
# 特征点检测器
sp = dlib.shape_predictor(predictor_path)

# 导入待处理图片
img = dlib.load_rgb_image(face_file_path)

# 人脸检测
dets = detector(img, 1)
# 人脸总数
num_faces = len(dets)
if num_faces == 0:
    print("Sorry, there were no faces found in '{}'".format(face_file_path))
    exit()

# 查找五个点的坐标
faces = dlib.full_object_detections()
for detection in dets:
    faces.append(sp(img, detection))

window = dlib.image_window()

# 特征点对齐：显示对齐之后的图片
images = dlib.get_face_chips(img, faces, size=320)
for image in images:
    window.set_image(image)
    dlib.hit_enter_to_continue()
