#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
#  author:cheney<XZCheney@gmail.com>
# 人脸识别

import cv2
import dlib

from PyQt5.QtCore import QTimer, QThread, pyqtSignal, QRegExp, Qt
from PyQt5.QtGui import QImage, QPixmap, QIcon, QTextCursor, QRegExpValidator
from PyQt5.QtWidgets import QDialog, QApplication, QMessageBox,QWidget
from PyQt5.uic import loadUi

import os
import logging
import logging.config
import sys
import threading
import queue
import numpy as np
import pandas as pd
import multiprocessing
import winsound

from configparser import ConfigParser
from datetime import datetime


class UI_face_reco(QWidget):
    receiveLogSignal = pyqtSignal(str)

    def __init__(self):
        super(UI_face_reco, self).__init__()
        loadUi('ui/face_reco.ui', self)
        self.setWindowIcon(QIcon('icons/icon.png'))
        self.setWindowTitle('人脸识别')
        self.setFixedSize(1040, 610)

        self.cap = cv2.VideoCapture()
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
        self.face_reco = dlib.face_recognition_model_v1("models/face_recognition_resnet_model_v1.dat")

        self.info_csv = "data\\info.csv"  # 人员信息
        self.features_csv = "data\\features.csv"  #人脸特征 从人脸图像中提取人脸特征
        self.features_known_arr = self.return_features_known_arr()

        self.font = cv2.FONT_HERSHEY_COMPLEX
        self.logQueue = queue.Queue()  # 日志队列

        # 摄像头
        self.isLocalCameraEnabled = False
        self.CheckBox_uselocalcamera.stateChanged.connect(self.use_local_camera)

        self.btn_opencam.clicked.connect(self.opencam)

        # 计时器
        self.timer_camera = QTimer(self)
        self.timer_camera.timeout.connect(self.show_camera)

        # 调试模式
        self.isDebugMode = False
        self.confidenceThreshold = 0.4
        self.CheckBox_debug.stateChanged.connect(self.enableDebug)
        self.SpinBox_Threshold.valueChanged.connect(self.setConfidenceThreshold)

        # 日志系统
        self.receiveLogSignal.connect(self.logOutput)
        self.logOutputThread = threading.Thread(target=self.receiveLog, daemon=True)
        self.logOutputThread.start()

    # 读取已知人脸特征数据
    def return_features_known_arr(self):
        if not os.path.isfile(self.info_csv):
            logging.error('系统找不到人员信息文件{}'.format(self.info_csv))
            self.logQueue.put('Error：系统找不到人员信息文件，请先录入相关人脸信息!')
        elif not os.path.isfile(self.features_csv):
            logging.error('系统找不到人脸特征{}'.format(self.features_csv))
            self.logQueue.put('Error：未找不到人脸特征文件，请先提取人脸特征!')
        else:
            self.features_reader = pd.read_csv(self.features_csv, header=None)
            self.info_reader = pd.read_csv(self.info_csv, header=None)

            features_known_arr = []
            for i in range(self.features_reader.shape[0]):
                features_someone_arr = []
                for j in range(0, len(self.features_reader.iloc[i, :])):
                    features_someone_arr.append(self.features_reader.iloc[i, :][j])
                features_known_arr.append(features_someone_arr)

            self.LcdNum_faces.display(len(features_known_arr))

        return features_known_arr

    # 是否使用本地摄像头
    def use_local_camera(self, state):
        if state == Qt.Checked:
            self.isLocalCameraEnabled = True
        else:
            self.isLocalCameraEnabled = False

    # 打开摄像头
    def opencam(self):
        if self.isLocalCameraEnabled:
            CAMNUM = 1
        else:
            CAMNUM = 0

        if self.timer_camera.isActive() == False:
            ok = self.cap.open(CAMNUM)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if not ok:
                logging.error('无法调用电脑摄像头{}'.format(CAMNUM))
                self.logQueue.put('Error：初始化摄像头失败')
                self.cap.release()
                self.btn_opencam.setIcon(QIcon('icons/error.png'))
            else:
                self.btn_opencam.setText(u'关闭摄像头')
                self.timer_camera.start(5)
                self.btn_opencam.setIcon(QIcon('icons/success.png'))
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.label_show_camera.clear()
            self.label_show_camera.setText(u'<font color=red>摄像头未开启</font>')
            self.btn_opencam.setText(u'打开摄像头')
            self.btn_opencam.setIcon(QIcon())

    # 显示图像
    def show_camera(self):
        ok, frame = self.cap.read()
        self.displayImage(frame)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faceRects = self.detector(frame_gray, 0)

        positions_list = []  # 人脸坐标
        names_list = []  # 姓名

        if len(faceRects) != 0:
            # 获取当前捕获到的图像的所有人脸的特征，存储到 features_arr
            features_arr = []
            for i in range(len(faceRects)):
                shape = self.predictor(frame, faceRects[i])
                features_arr.append(self.face_reco.compute_face_descriptor(frame, shape))
                names_list.append("unknown")

                # 每个捕获人脸的名字坐标
                positions_list.append(tuple(
                        [faceRects[i].left(), int(faceRects[i].bottom() + (faceRects[i].bottom() - faceRects[i].top()) / 4)]))

                e_distance_list = []
                for k in range(len(self.features_known_arr)):
                    if str(self.features_known_arr[k][0]) != '0.0':
                        e_distance = self.return_euclidean_distance(features_arr[i], self.features_known_arr[k])
                        e_distance_list.append(e_distance)
                    else:
                        e_distance_list.append(999999999)
                the_most_similar_person = e_distance_list.index(min(e_distance_list))

                if min(e_distance_list) < self.confidenceThreshold :
                    names_list[i] = self.info_reader.iloc[the_most_similar_person, 0].split('_')[0]
                    cv2.putText(frame, names_list[i], positions_list[i], self.font, 1, (0, 255, 255), 1, cv2.LINE_AA)
                    self.logQueue.put('人脸识别成功，欢迎{}！'.format(names_list[i]))
                else:
                    cv2.putText(frame, names_list[i], positions_list[i], self.font, 1, (0, 255, 255), 1, cv2.LINE_AA)
                    logging.error('Error:{}尝试非法闯入！'.format(names_list[i]))
                    self.logQueue.put('Error:{}尝试非法闯入！'.format(names_list[i]))

            for faceRect in faceRects:
                x = faceRect.left()
                y = faceRect.top()
                w = faceRect.right() - faceRect.left()
                h = faceRect.bottom() - faceRect.top()
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 255), 2)
            self.displayImage(frame)
        # else:
        #     self.displayImage(frame)

    def displayImage(self, img):
        img = cv2.resize(img, (640, 480))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ShowImage = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
        self.label_show_camera.setPixmap(QPixmap.fromImage(ShowImage))

    # 计算两个128D向量间的欧式距离
    def return_euclidean_distance(self,feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    # 是否开启调试模式
    def enableDebug(self,state):
        if state == Qt.Checked:
            self.isDebugMode = True
            self.logQueue.put('Debug模式已开启！')
        else:
            self.isDebugMode = False
            self.logQueue.put('Debug模式已关闭！')

    # 设置置信度阈值
    def setConfidenceThreshold(self):
        if self.isDebugMode:
            self.confidenceThreshold = self.SpinBox_Threshold.value()
            self.logQueue.put('当前置信度阈值为{}！'.format(self.confidenceThreshold))


    # 系统日志服务常驻，接收并处理系统日志
    def receiveLog(self):
        while True:
            data = self.logQueue.get()
            if data:
                self.receiveLogSignal.emit(data)
            else:
                continue

    #  LOG输出
    def logOutput(self, log):
        # 获取当前系统时间
        time = datetime.now().strftime('[%Y/%m/%d %H:%M:%S]')
        log = time + ' ' + log + '\n'
        self.TextEdit_log.moveCursor(QTextCursor.End)
        self.TextEdit_log.insertPlainText(log)
        self.TextEdit_log.ensureCursorVisible()  # 自动滚屏

    # 窗口关闭事件，关闭定时器、摄像头
    def closeEvent(self, event):
        if self.timer_camera.isActive():
            self.timer_camera.stop()

        if self.cap.isOpened():
            self.cap.release()
        event.accept()

if __name__ == '__main__':
    logging.config.fileConfig('config/logging.conf')
    app = QApplication(sys.argv)
    w = UI_face_reco()
    w.show()
    sys.exit(app.exec())















































# import cv2
# import dlib
# import numpy as np
# import pandas as pd
#
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('models\\shape_predictor_68_face_landmarks.dat')
# face_reco = dlib.face_recognition_model_v1("models\\face_recognition_resnet_model_v1.dat")
#
#
# # 计算两个128D向量间的欧式距离
# def return_euclidean_distance(feature_1, feature_2):
#     feature_1 = np.array(feature_1)
#     feature_2 = np.array(feature_2)
#     dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
#     return dist
#
# features_csv = "data\\features.csv"
# info_csv = "data\\info.csv"
# features_reader = pd.read_csv(features_csv, header=None)
# info_reader = pd.read_csv(info_csv, header=None)
#
# font = cv2.FONT_HERSHEY_COMPLEX
# # 读取已知人脸数据
# features_known_arr = []
#
# for i in range(features_reader.shape[0]):
#     features_someone_arr = []
#     for j in range(0, len(features_reader.iloc[i, :])):
#         features_someone_arr.append(features_reader.iloc[i, :][j])
#     features_known_arr.append(features_someone_arr)
# print("Faces in Database：", len(features_known_arr))
#
# cap = cv2.VideoCapture(0)
#
# while cap.isOpened():
#     ok, frame = cap.read()
#     if not ok:
#         break
#
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#
#     faceRects = detector(frame_gray, 0)
#
#     positions_list = []
#     names_list = []
#
#     if len(faceRects) != 0:
#         # 获取当前捕获到的图像的所有人脸的特征，存储到 features_arr
#         features_arr = []
#         for i in range(len(faceRects)):
#             shape = predictor(frame, faceRects[i])
#             features_arr.append(face_reco.compute_face_descriptor(frame, shape))
#
#             names_list.append("unknown")
#
#             # 每个捕获人脸的名字坐标
#             positions_list.append(tuple(
#                 [faceRects[i].left(), int(faceRects[i].bottom() + (faceRects[i].bottom() - faceRects[i].top()) / 4)]))
#
#             e_distance_list = []
#             for k in range(len(features_known_arr)):
#                 if str(features_known_arr[k][0]) != '0.0':
#                     e_distance = return_euclidean_distance(features_arr[i], features_known_arr[k])
#                     e_distance_list.append(e_distance)
#                 else:
#                     e_distance_list.append(999999999)
#             the_most_similar_person = e_distance_list.index(min(e_distance_list))
#
#             if min(e_distance_list) < 0.4:
#                 names_list[i] = info_reader.iloc[the_most_similar_person, 0].split('_')[0]
#                 cv2.putText(frame, names_list[i], positions_list[i], font, 1, (0, 255, 255), 1, cv2.LINE_AA)
#                 cv2.putText(frame, "faces: " + str(len(faceRects)), (20, 100), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
#             else:
#                 cv2.putText(frame, names_list[i], positions_list[i], font, 1, (0, 255, 255), 1, cv2.LINE_AA)
#
#         for faceRect in faceRects:
#             x = faceRect.left()
#             y = faceRect.top()
#             w = faceRect.right() - faceRect.left()
#             h = faceRect.bottom() - faceRect.top()
#             cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 255), 2)
#
#     cv2.imshow("Face Recognition", frame)
#     if cv2.waitKey(1) & 0xff == ord('q'):
#         break
#
# cv2.destroyWindow("Face Recognition")
# cap.release()