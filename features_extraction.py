#!/usr/bin/python3
# conding=utf8
#  author:cheney<XZCheney@gmail.com>
# 从人脸图像文件中提取人脸特征存入 CSV

import os
import cv2
import csv
import dlib
from skimage import io
import numpy as np

root = "data\\facesdata"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models\\shape_predictor_68_face_landmarks.dat")
face_reco = dlib.face_recognition_model_v1("models\\face_recognition_resnet_model_v1.dat")


def return_128d_features(path_img):
    img = io.imread(path_img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector(img_gray, 1)
    if len(faces) != 0:
        shape = predictor(img_gray, faces[0])
        face_descriptor = face_reco.compute_face_descriptor(img_gray, shape)
    else:
        face_descriptor = 0
        print("人脸数据为空！")
    return face_descriptor

# 将文件夹中照片特征提取出来, 写入 CSV
def return_features_mean(person_face_dir):
    features_list = []
    faces_list = os.listdir(person_face_dir)
    if faces_list:
        for i in range(len(faces_list)):
            features_128d = return_128d_features(os.path.join(person_face_dir, faces_list[i]))
            # 遇到没有检测出人脸的图片跳过
            if features_128d == 0:
                i += 1
            else:
                features_list.append(features_128d)
    else:
        print("Warning: No images in " + person_face_dir)

    if features_list:
        features_mean = np.array(features_list).mean(axis=0)
    else:
        features_mean = '0'

    return features_mean


if __name__ == '__main__':
    with open("data\\features.csv", "w", newline="") as features_file:
        with open("data\\info.csv", "w", newline="") as info_file:
            features_writer = csv.writer(features_file)
            info_writer = csv.writer(info_file)
            for path, dirs, files in os.walk(root):
                for dir_name in dirs:
                    features_mean = return_features_mean(os.path.join(root, dir_name))
                    info_writer.writerow([dir_name])
                    features_writer.writerow(features_mean)
    print("所有人脸数据录入完毕！")