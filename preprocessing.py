#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:28:13 2019

@author: jarvis
"""

# Multi-proocess and multi-gpu

import mxnet as mx
from scipy.io import loadmat
from datetime import datetime
import pandas as pd
import numpy as np
import cv2

import threading
from queue import Queue

from mx_mtcnn.mtcnn_detector import MtcnnDetector
from pose import get_rotation_angle


global POISONPILL
POISONPILL = False

global DISTRIBUTOR_ACK
DISTRIBUTOR_ACK = False

#COLUMS = ["age", "gender", "image", "org_box", "trible_box", "landmarks", "roll", "yaw", "pitch"]


def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))
    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


def gen_boundbox(box, landmark):
    # gen trible boundbox
    ymin, xmin, ymax, xmax = map(int, [box[1], box[0], box[3], box[2]])
    w, h = xmax - xmin, ymax - ymin
    nose_x, nose_y = (landmark[2], landmark[2+5])
    w_h_margin = abs(w - h)
    top2nose = nose_y - ymin
    return np.array([
        [(xmin - w_h_margin, ymin - w_h_margin), (xmax + w_h_margin, ymax + w_h_margin)],  # inner
        [(nose_x - top2nose, nose_y - top2nose), (nose_x+top2nose, nose_y + top2nose)],  # middle
        [(nose_x - w//2, nose_y - w//2), (nose_x + w//2, nose_y + w//2)]  # outer box
    ])


def sample_consumer(gpu_id, cpu_id=0, use_gpu=True):
    if use_gpu:
        detector = MtcnnDetector(model_folder=None, ctx=mx.gpu(gpu_id), num_worker=1, minsize=50, accurate_landmark=True)
    else:
        detector = MtcnnDetector(model_folder=None, ctx=mx.cpu(cpu_id), num_worker=1, minsize=50, accurate_landmark=True)
    
    # image_path = os.path.join(self.data_dir, series.full_path[0])
    while True:
        obj = dQ.get()
        image_path = obj['full_path']
        try:
            print(image_path)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if not np.isnan(obj['second_face_score']):
                raise Exception("more than one face~---%s"%(image_path))
            
            ret = detector.detect_face(image) 
            if not ret:
                raise Exception("cant detect facei: %s"%image_path)
            bounds, lmarks = ret
            if len(bounds) > 1:
                raise Exception("more than one face %s"%image_path)
            
            crops = detector.extract_image_chips(image, lmarks, padding=0.4)  # aligned face with padding 0.4 in papper
        
            if len(crops) == 0:
                raise Exception("no crops~~ %s"%(image_path))
            if len(crops) > 1:
                raise Exception("more than one face~---%s"%(image_path))
            
            ret = detector.detect_face(crops[0]) 
            if not ret:
                raise Exception("cant detect facei: %s"%image_path)
            bounds, lmarks = ret
            if len(bounds) > 1:
                raise Exception("more than one face %s"%image_path)
            
            org_box, first_lmarks = bounds[0], lmarks[0]
            trible_box = gen_boundbox(org_box, first_lmarks)
            pitch, yaw, roll = get_rotation_angle(crops[0], first_lmarks) # gen face rotation for filtering
            image = crops[0]   # select the first align face and replace
        
            series = {}
            series["age"] = obj["age"]
            series["gender"] = obj["gender"]

            status, buf = cv2.imencode(".jpg", image)
            series["image"] = buf.tostring() 
            series["org_box"] = org_box.dumps()  # xmin, ymin, xmax, ymax
            series["landmarks"] = first_lmarks.dumps()  # y1..y5, x1..x5
            series["trible_box"] = trible_box.dumps() 
            series["yaw"] = yaw
            series["pitch"] = pitch
            series["roll"] = roll
            
            collectQ.put(series)
        except Exception as ee:
            print(ee)
            pass
    ackQ.put('ok')

def distributor(mat_path = 'dataset/wiki_crop/wiki.mat', name='wiki', nums=-1):
    meta = loadmat(mat_path)
    full_path = meta[name][0, 0]["full_path"][0][:nums]
    dob = meta[name][0, 0]["dob"][0][:nums]  # Matlab serial date number
    mat_gender = meta[name][0, 0]["gender"][0][:nums]
    photo_taken = meta[name][0, 0]["photo_taken"][0][:nums]  # year
    face_score = meta[name][0, 0]["face_score"][0][:nums]
    second_face_score = meta[name][0, 0]["second_face_score"][0][:nums]
    mat_age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]
    dataset = pd.DataFrame({"full_path": full_path, "age": mat_age, "gender": mat_gender, "second_face_score": second_face_score, "face_score": face_score})
    dataset = dataset[(dataset.age >= 0) & (dataset.age <= 100)]
    dataset = dataset[dataset.gender != np.nan]
    dataset = dataset[np.isnan(dataset.second_face_score)]
    
    for i in range(len(dataset)):
        sample = dict(dataset.iloc[i])
        sample['full_path'] = 'dataset/wiki_crop/' + sample['full_path'][0]
        dQ.put(sample)
    
    
if __name__ == '__main__':
    dQ = Queue()
    collectQ = Queue()
    ackQ = Queue
    
    threads = []
    process_threads = 4
    for i in range(process_threads):
        threads.append(threading.Thread(target=sample_consumer, args=(i,)))

    process_threads = 4
    for i in range(process_threads):
        threads.append(threading.Thread(target=sample_consumer, args=(0, i, False)))
    
    dThread = threading.Thread(target=distributor, args=('dataset/wiki_crop/wiki.mat', 'wiki', -1))
    dThread.start()
    
    for i in range(len(threads)):
        threads[i].start()
    
    #Thread watcher
    pickle_list = []
    ack_vars = []
    while True:
        if collectQ.empty():
            continue
        obj = collectQ.get()
        pickle_list.append(obj)
        if ackQ.empty():
            continue
        ack_vars.append(ackQ.get())
        if len(ack_vars)==len(threads):
            break
        
        

    
    import pickle
    pickle_df = pd.DataFrame(pickle_list)
    PIK = "pickle_df.dat"
    with open(PIK, "wb") as f:
        pickle.dump(pickle_df, f)
        

