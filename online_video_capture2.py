# 实时：视频图像采集(dlib)
import cv2
import dlib

cap = cv2.VideoCapture(0)
win = dlib.image_window()
win.set_title("VideoCapture")
# 从视频流循环帧
while cap.isOpened():
    ret, frame = cap.read()
    if frame is None:
        break
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    win.clear_overlay()
    win.set_image(img)
cap.release()
