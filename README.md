# Introduce
Face Check-in System Based on Dlib  
Face check-in system is implemented based on machine learning/deep learning framework Dlib and OpenCV library.  
Experimental platform: Pycharm.  
This project uses Python for modular design.  
The main functions are as follows:  
1. Real-time video acquisition program design. There are two schemes: 1. opencv 2. opencv + dlib. These two schemes are the difference of windows.  
2. The real-time face detection program (using frontal_face_dtector interface) is based on five feature points.  
3. The real-time feature point calibration program (using shape_predictor interface) is based on 68 feature points.  
4. Programming of real-time feature point alignment (interface: get_face_chip).  
5. Create a face database + real-time face comparison (face_recognition library is used for encoding).  
This includes: (1) making a face registry; (2) real-time comparison based on video stream, and displaying the names of registered faces.
6. Real-time blinking behavior recognition.  
Intelligent judgment is added to prevent fraud:  
The first step is to check whether the aspect ratio of the eye is lower than our blink threshold. If so, we increment the number of consecutive frames indicating that blinking is occurring (that is, the previous blink threshold, set here to 2, which means that the blink speed should not be too fast). Otherwise, we will deal with the case where the aspect ratio of the eye is not lower than the blink threshold. We will check it to see if there are enough consecutive frames containing the blink rate below our predefined threshold. If the check passes, we increase the total number of flickers. Then we reset the COUNTER number of successive flickers.  
      
At the same time, it provides the code of off-line comparison, which needs to be improved. It combines blink recognition with real-time face comparison.

# Chinese instructions
基于Dlib的人脸签到系统  
基于机器学习/深度学习框架Dlib、OpenCV库实现人脸签到系统。  
实验平台：Pycharm。  
本项目采用python进行模块化设计。  
主要功能如下：  
1，实时视频采集程序设计，两种方案：①opencv ②opencv+dlib，这两个就是窗口的差异。  
2，实时人脸检测程序设计（用到frontal_face_dtector接口）基于5特征点的。  
3，实时特征点标定程序设计（用到shape_predictor接口）基于68特征点。  
4，实时特征点对齐程序设计（接口：get_face_chip）。  
5，创建人脸库+实时人脸比对（用到face_recognition库，用来encoding）。  
   这里包含①人脸注册库制作 ②基于视频流的实时比对，并且显示已注册人脸的名字。  
6，实时眨眼行为识别。  
   加入了智能判断防止欺诈：  
第一步检查眼睛纵横比是否低于我们的眨眼阈值，如果是，我们递增指示正在发生眨眼的连续帧数（即前面的闪烁阀值，这里设定为2，就是要求眨眼的速度不能过快）。否则，我们将处理眼高宽比不低于眨眼阈值的情况，我们对其进行检查，看看是否有足够数量的连续帧包含低于我们预先定义的阈值的眨眼率。如果检查通过，我们增加总的闪烁次数。然后我们重新设置连续闪烁次数 COUNTER。  

同时提供了离线比对的代码，有待提高的地方，将眨眼识别与实时人脸比对结合。
