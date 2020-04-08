import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2

import os
import numpy as np
import cv2
import xml.etree.cElementTree as ET
import copy
import matplotlib.pyplot as plt
import matplotlib
import math
import h5py
import json
import shutil

# Added kalman filter into the system
class KalmanFilter(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
            (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

def calculate_kalman(speed_temp):
    dt = 1.0/60
    # dt = 1.0/60
    F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
    H = np.array([1, 0, 0]).reshape(1, 3)
    Q = np.array([[0.005, 0.00, 0.0], [0.005, 0.005, 0.0], [0.0, 0.0, 0.0]])
    R = np.array([0.5]).reshape(1, 1)

    kf = KalmanFilter(F = F, H = H, Q = Q, R = R)


    # all_speed = []
    # for l in range(len(speed_temp)):
    #     all_speed += speed_temp[l]

    #all_speed = all_speed.astype('float64')
    #all_speed = np.array(all_speed, dtype=np.float64)

    predictions = []
    for z in speed_temp:
        predictions.append(np.dot(H,  kf.predict())[0])
        kf.update(z)

    return predictions
    
def filter_speed(data_dir):
    speed_path = os.path.join(data_dir,"speed.txt")

    with open(speed_path) as speed:
        speed_info = speed.readlines()

    # Copy a file
    # src_dir = data_dir+"speed.txt"
    # dst_dir = data_dir+"speed_filtered.txt"
    # shutil.copy(src_dir,dst_dir)

    # ########################
    save = os.path.join(data_dir, 'speed_anjay.txt')
    position = open(save, 'w+')

    # Create raw
    for i in range(len(speed_info)):
        speed_read = speed_info[i].strip().split()
        frame = speed_read[0]
        position.write(frame)
        position.write("\n")
    position.close()

    # Process for the replaced file
    speed2_path = os.path.join(data_dir,"speed_anjay.txt")

    with open(speed2_path, "r+") as speeds:
        speed2_info = speeds.readlines()

    print("len = ", len(speed2_info))

    # Scan oll of the object ID
    list_obj = []
    for i in range(len(speed_info)):
        speed_read = speed_info[i].strip().split()

        each_obj_speed = speed_read[1:]
        object_id = each_obj_speed[::2]
        object_speed = each_obj_speed[1::2]

        for j in range(len(object_id)):
            objek = object_id[j]
            list_obj.append(objek)

        list_obj = list(dict.fromkeys(list_obj))

    print("list_obj = ", list_obj)

    # Save the speed in a list
    # First scan over the object
    for i in range(len(list_obj)):
        speed_temp = []
        frame_temp = []
        query_temp = []

        # Scan over the speed.txt
        for j in range(len(speed_info)):
            speed_read = speed_info[j].strip().split()

            frame = speed_read[0]
            each_obj_speed = speed_read[1:]
            object_id = each_obj_speed[::2]
            object_speed = each_obj_speed[1::2]

            # Scan over the each frame
            for k in range(len(object_id)):
                objek = object_id[k]
                kecepatan = object_speed[k]

                # Kalo sama save di suatu list
                if objek == list_obj[i]:
                    speed_temp.append(float(kecepatan))
                    frame_temp.append(frame)
                    query_temp.append(j)

        predictions = calculate_kalman(speed_temp)

        id_now = list_obj[i]

        speed2_path = os.path.join(data_dir,"speed_anjay.txt")
        speeds = open(speed2_path, 'r+')
        speed2_info = speeds.readlines()

        for x in range(len(speed2_info)):
            timpa = speed2_info[x].strip()
            tag = False
            #print("timpa = ", timpa)

            for y in range(len(query_temp)):
                urutan = query_temp[y]
                speed_now = predictions[y]

                if x == urutan:
                    tag = True
                    kec = '%.2f '%speed_now[0]
                    baru_tulis = timpa + " " + str(id_now) + " " + str(kec)
                    speeds.write(baru_tulis)
                    
            if tag == False:
                speeds.write(timpa)

            speeds.write("\n")

        speeds.close()

        # Take only the new
        n = len(speed2_info)
        nfirstlines = []

        with open(speed2_path) as f, open(os.path.join(data_dir,"bigfiletmp.txt"), "w") as out:
            for x in range(n):
                nfirstlines.append(next(f))
            for line in f:
                out.write(line)

        # NB : it seems that `os.rename()` complains on some systems
        # if the destination file already exists.
        os.remove(speed2_path)
        os.rename(os.path.join(data_dir,"bigfiletmp.txt"), speed2_path)
