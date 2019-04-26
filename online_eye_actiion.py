# 实时活体检测：眨眼行为识别
import scipy
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2


def eye_aspect_ratio(eye):
    # 计算两个集合之间的欧几里得距离
    # 垂直眼标志（X，Y）坐标
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # 计算水平之间的欧几里得距离
    # 水平眼标志（X，Y）坐标
    C = dist.euclidean(eye[0], eye[3])
    # 眼睛长宽比的计算
    ear = (A + B) / (2.0 * C)
    # 返回眼镜的长宽比
    return ear


# 定义两个常数
# 眼睛长宽比
# 闪烁阈值
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 2
# 初始化帧计数器和眨眼总数
COUNTER = 0
TOTAL = 0

# 初始化DLIB的人脸检测器（HOG），然后创建面部标志物预测
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('model//shape_predictor_68_face_landmarks.dat')

# 分别获取左右眼面部标志的索引
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

cap = cv2.VideoCapture(0)

# 从视频流循环帧
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 检测灰度帧中的人脸
    rects = detector(gray, 0)

    # 人脸检测循环
    for rect in rects:
        # 确定脸部区域的面部标志，然后将面部标志（x，y）坐标转换为数字阵列
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # 提取左眼和右眼坐标，然后使用坐标计算双眼的眼睛长宽比
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # 双眼平均长宽比
        ear = (leftEAR + rightEAR) / 2.0

        # 计算左眼和右眼的标志点并绘制
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # 在图片中标注人脸，并显示
        left = rect.left()
        top = rect.top()
        right = rect.right()
        bottom = rect.bottom()
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

        '''
        第一步检查眼睛纵横比是否低于我们的眨眼阈值，如果是，我们递增指示正在发生眨眼的连续帧数。
        否则，我们将处理眼高宽比不低于眨眼阈值的情况，我们对其进行检查，
        看看是否有足够数量的连续帧包含低于我们预先定义的阈值的眨眼率。
        如果检查通过，我们增加总的闪烁次数。然后我们重新设置连续闪烁次数 COUNTER。
        '''
        if ear < EYE_AR_THRESH:
            COUNTER += 1
        else:
            # 如果眼睛闭合足够数量，那么眨眼总数增加
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            # 重置眼帧计数器
            COUNTER = 0

        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        cv2.putText(frame, "COUNTER: {}".format(COUNTER), (150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    print(len(rects))
    # if the `q` key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# do a bit of cleanup
cv2.destroyAllWindows()