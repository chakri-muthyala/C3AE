# Multi-proocess and multi-gpu
import time
import pandas as pd
import numpy as np
import cv2
import json

import threading
from queue import Queue

import os
import sys

global POISONPILL
POISONPILL = False

def sample_consumer(id_):
    tid=id_
    pitch=True
    while pitch:
        if POISONPILL:
            break
        if not dQ.empty():
            couple = dQ.get()
            if type(couple)==type('jc'):
                if couple=='ACK':
                    break
            img_path, json_path = couple
            try:
                img = cv2.imread(img_path)
                with open(json_path, 'r') as f:
                    jdict = json.load(f)
                    labels = jdict["annotations"]
            except Exception as e:
                print(e)
                
                continue
            for label in labels:
                try:
                    sample = label.copy()
                    x = sample["boxx"]
                    x = max(x, 0)
                    y = sample["boxy"]
                    y = max(y, 0)
                    w = sample["boxw"]
                    h = sample["boxh"]
                    m = min(h, w) #min to maintain aspect ratio
                    img_crop = img[y:y+m, x:x+m, :]
                    #print('img-shape',img.shape)
                    img_crop = cv2.resize(img_crop, (256, 256))
                    status, buf = cv2.imencode(".jpg", img_crop)
                    sample["image"] = buf.tostring() 
                    sample.pop('e_null')
                    collectQ.put(sample)
                except Exception as e:
                    print(e)
                    print(img_path, json_path)
                    print('log:, img_shape', img.shape)
                    print('log:, img_None', img is None)
                    print('log:, img_coords', (x,y,w,h))
                    print('--------------O----------------')
                    
                #print('sent to collectq')
    ackQ.put('ok')
    print('Sampler DONE')

def distributor(data_dir='data/'):
    data_dir='data/'
    root_dir = os.listdir(data_dir)
    root_dir.remove('MORPH.dat') #Excluders
    root_dir.remove('IMDB.dat')
    all_file_paths = []
    for sub_dir in root_dir:
        files_dir = os.path.join(data_dir, sub_dir)
        file_dirs = os.listdir(files_dir)
        for file_dir in file_dirs:
            file = os.path.join(files_dir, file_dir)
            if file[-4:] == 'json':
                continue            
            all_file_paths.append(file)
    #send couple
    for png in all_file_paths:
        img_path = png
        json_path = img_path[:-3]+'json'
        dQ.put((img_path, json_path))
    while True: ## Assuming 100 threads at max
        if POISONPILL:
            break
        v = 'ACK'
        dQ.put(v)
        time.sleep(10)
    print('distributor DONE')

    
if __name__ == '__main__':
    dQ = Queue()
    collectQ = Queue()
    ackQ = Queue()

    threads = []
    process_threads = 30 # for no. of cpus 
    for i in range(process_threads):
        threads.append(threading.Thread(target=sample_consumer, args=(i,)))

    for i in range(len(threads)):
        threads[i].daemon = True
        threads[i].start()

    dThread = threading.Thread(target=distributor, args=('data/',))
    dThread.daemon = True
    dThread.start()

    #Thread watcher
    pickle_list = []
    ack_vars = []
    while True:
        try:
            if not collectQ.empty():
                obj = collectQ.get()
                pickle_list.append(obj)
            if not ackQ.empty():
                awr = ackQ.get()
                ack_vars.append(awr)
            if len(ack_vars)==len(threads):
                print('broken by len of threads')
                break
    #            print('len_ack_vars', len(ack_vars))
    #            print('pickle_list_count', len(pickle_list))
        except Exception as e:
            print(e)

    import pickle
    pickle_df = pd.DataFrame(pickle_list)
    PIK = "pickle_df_ake.dat"
    with open(PIK, "wb") as f:
        pickle.dump(pickle_df, f)

    POISONPILL = True
    print('DONE!!')


