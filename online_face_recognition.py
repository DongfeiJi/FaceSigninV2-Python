import face_recognition
import cv2

video_capture = cv2.VideoCapture(0)

# 01:导入已注册人脸图像，并向量化表示
image1 = face_recognition.load_image_file("./data/login/004.jpg")
image1_face_encoding = face_recognition.face_encodings(image1)[0]

image2 = face_recognition.load_image_file("./data/login/002.jpg")
image2_face_encoding = face_recognition.face_encodings(image2)[0]

image3 = face_recognition.load_image_file("./data/login/006.jpg")
image3_face_encoding = face_recognition.face_encodings(image3)[0]

image4 = face_recognition.load_image_file("./data/login/008.jpg")
image4_face_encoding = face_recognition.face_encodings(image4)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    image1_face_encoding,
    image2_face_encoding,
    image3_face_encoding
]
known_face_names = [
    "zhangzl",
    "sunli",
    "DongfeiJi",
    "JingyanHan"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # 捕获视频流
    ret, frame = video_capture.read()
    # 尺寸重置
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # BCR->RGB
    rgb_small_frame = small_frame[:, :, ::-1]
    # Only process every other frame of video to save time
    if process_this_frame:
        # 查找和压缩当前帧
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        # 与已注册人脸数据对比
        for face_encoding in face_encodings:
            # 人脸对比
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            # 匹配到合适的目标，则显示姓名
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            face_names.append(name)

    process_this_frame = not process_this_frame

    # 显示识别结果
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        # 绘制矩形
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # 显示标签
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    # 显示识别结果
    cv2.imshow('Result:', frame)
    # 等待用户退出指令
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 资源释放
video_capture.release()
cv2.destroyAllWindows()