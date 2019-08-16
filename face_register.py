#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
#  author:cheney<XZCheney@gmail.com>
import os
import cv2
import sys
import dlib
import queue
import logging
import logging.config
import threading
from datetime import datetime

from PyQt5.QtCore import QTimer, pyqtSignal,Qt,QRegExp
from PyQt5.QtGui import QImage, QPixmap, QIcon, QTextCursor,QRegExpValidator
from PyQt5.QtWidgets import QWidget,QApplication
from PyQt5.uic import loadUi


class UI_face_register(QWidget):
    receiveLogSignal = pyqtSignal(str)

    def __init__(self):
        super(UI_face_register, self).__init__()
        loadUi('ui/face_register.ui', self)
        self.setWindowIcon(QIcon('icons/icon.png'))
        self.setWindowTitle('人脸注册')
        self.setFixedSize(1040, 600)
        self.cnt = 0
        self.cap = cv2.VideoCapture()
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
        self.isFaceInfoReady = False
        self.logQueue = queue.Queue()  # 日志队列

        # 新建人脸文件夹
        self.btn_newfacefolder.clicked.connect(self.new_face_folder)

        # 摄像头
        self.isLocalCameraEnabled = False
        self.CheckBox_uselocalcamera.stateChanged.connect(self.use_local_camera)

        self.btn_opencam.clicked.connect(self.opencam)

        # 计时器
        self.timer_camera = QTimer(self)
        self.timer_camera.timeout.connect(self.show_camera)

        # 人脸检测
        self.isFaceDetectEnabled = False
        self.btn_enableFaceDetect.clicked.connect(self.enableFaceDetect)

        # 日志系统
        self.receiveLogSignal.connect(self.logOutput)
        self.logOutputThread = threading.Thread(target=self.receiveLog, daemon=True)
        self.logOutputThread.start()

# 新建人脸文件夹
    def new_face_folder(self):
        root = "data\\facesdata"
        if os.path.isdir(root):
            pass
        else:
            os.mkdir(root)

        # 使用正则表达式限制用户输入
        gh_regx = QRegExp('^[0-9]{12}$')
        gh_validator = QRegExpValidator(gh_regx, self.Edit_input_gh)
        self.Edit_input_gh.setValidator(gh_validator)

        xm_regx = QRegExp('^[\u4e00-\u9fa5]{1,10}$')
        xm_validator = QRegExpValidator(xm_regx, self.Edit_input_xm)
        self.Edit_input_xm.setValidator(xm_validator)

        self.face_id = self.Edit_input_gh.text()
        self.face_name = self.Edit_input_xm.text()

        if self.face_name == "":
            logging.error('Error：姓名不能为空')
            self.logQueue.put('Error：姓名不能为空')
            self.Edit_input_xm.setFocus()
            return
        elif self.face_id == "":
            logging.error('Error：工号不能为空')
            self.logQueue.put('Error：工号不能为空')
            self.Edit_input_gh.setFocus()
            return
        else:
            face_info = self.face_name + "_" + self.face_id
            self.person_face_dir = os.path.join(root, face_info)

            if os.path.isdir(self.person_face_dir):
                pass
            else:
                os.mkdir(self.person_face_dir)
                self.isFaceInfoReady = True
                self.btn_newfacefolder.setIcon(QIcon('icons/success.png'))
                self.logQueue.put("新建"+self.person_face_dir+"文件夹成功")

# 是否使用本地摄像头
    def use_local_camera(self,state):
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
                self.btn_enableFaceDetect.setEnabled(False)
                self.btn_opencam.setIcon(QIcon('icons/error.png'))
            else:
                self.btn_opencam.setText(u'关闭摄像头')
                self.btn_enableFaceDetect.setEnabled(True)
                self.timer_camera.start(5)
                self.btn_opencam.setIcon(QIcon('icons/success.png'))
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.label_show_camera.clear()
            self.label_show_camera.setText(u'<font color=red>摄像头未开启</font>')
            self.btn_opencam.setText(u'打开摄像头')
            self.btn_opencam.setIcon(QIcon())
            self.btn_enableFaceDetect.setEnabled(False)

    #  显示画面
    def show_camera(self):
        ok, self.frame = self.cap.read()
        self.displayImage(self.frame)
        if self.isFaceDetectEnabled:
            img_gray = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
            faceRects = self.detector(img_gray, 0)
            if len(faceRects) != 0:
                for faceRect in faceRects:
                    self.cnt += 1
                    x = faceRect.left()
                    y = faceRect.top()
                    w = faceRect.right() - faceRect.left()
                    h = faceRect.bottom() - faceRect.top()
                    image = self.frame[y - 10: y + h + 10, x - 10: x + w + 10]
                    cv2.imwrite(self.person_face_dir + "/face_" + str(self.cnt) + ".jpg", image)
                    cv2.rectangle(self.frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 2)

                if self.cnt > (9):
                    self.btn_enableFaceDetect.setIcon(QIcon())
                    self.btn_enableFaceDetect.setText('开启人脸检测')
                    self.btn_enableFaceDetect.setEnabled(False)
                    self.isFaceDetectEnabled = False
                    self.logQueue.put("Success："+"工号"+self.face_id+","+"姓名"+self.face_name+","+"您的人脸数据录入完成！")

            self.btn_enableFaceDetect.setIcon(QIcon('icons/success.png'))
            self.displayImage(self.frame)
        else:
            self.displayImage(self.frame)

    def displayImage(self, img):
        img = cv2.resize(img, (640, 480))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ShowImage = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
        self.label_show_camera.setPixmap(QPixmap.fromImage(ShowImage))

    def enableFaceDetect(self, status):
        if self.isFaceInfoReady:
            self.btn_enableFaceDetect.setText('关闭人脸检测')
            self.btn_enableFaceDetect.setIcon(QIcon('icons/success.png'))
            self.isFaceDetectEnabled = True
        else:
            self.btn_enableFaceDetect.setText('开启人脸检测')
            self.logQueue.put("Warning：请先录入人员信息后重试")
            self.btn_enableFaceDetect.setIcon(QIcon('icons/error.png'))
            self.isFaceDetectEnabled = False

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
    w = UI_face_register()
    w.show()
    sys.exit(app.exec())