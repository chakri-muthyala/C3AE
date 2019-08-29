#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 15:33:59 2019

@author: chakri
"""

import numpy as np
import cv2
import random


def image_transform(row, seed=100, contrast=(0.5, 2.5), bright=(-50, 50), rotation=(-15, 15), dropout=0., shape=(64, 64), is_training=True): 
    # (idx, row) = row[0], row[1]
    # idx = row.name
    img = np.fromstring(row["image"], np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    #cv2.imwrite("%s_%s__.jpg"%(row.age, row.gender), img)

    if is_training:
        img = random_erasing(img, dropout)

    cascad_imgs, padding = [], 0
    new_bd_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
    height, width = img.shape[:2]
    for bbox in np.loads(row.trible_box):
        h_min, w_min = bbox[0]
        h_max, w_max = bbox[1]
        #cv2.rectangle(img, (h_min, w_min), (h_max, w_max), (0,0,255), 2)
        cascad_imgs.append(cv2.resize(new_bd_img[max(w_min+padding, 0):min(w_max+padding, width), max(h_min+padding, 0): min(h_max+padding, height)], shape))
    ## if you want check data, and then you can remove these marks
    #if idx > 10000:
    #    cv2.imwrite("%s_%s_%s.jpg"%(row.age, row.gender, idx), cascad_imgs[2])
    if is_training:
       flag = random.randint(0, 3)
       imenf = []
       for imset in cascad_imgs:
           imenf.append(image_enforcing(imset, flag, contrast, bright, rotation))
    #    cascad_imgs = map(lambda x: image_enforcing(x, flag, contrast, bright, rotation), cascad_imgs)
    # return cascad_imgs
    return imenf

# No deps
def random_erasing(img, drop_out=0.3, aspect=(0.5, 2), area=(0.06, 0.10)):
    # https://arxiv.org/pdf/1708.04896.pdf
    if 1 - random.random() > drop_out:
        return img
    img = img.copy()
    height, width = img.shape[:-1]
    aspect_ratio = np.random.uniform(*aspect) 
    area_ratio = np.random.uniform(*area)
    img_area = height * width * area_ratio
    dwidth, dheight = np.sqrt(img_area * aspect_ratio), np.sqrt(img_area * 1 / aspect_ratio)  
    xmin = random.randint(0, height) 
    ymin = random.randint(0, width)
    xmax, ymax = min(height, int(xmin + dheight)), min(width, int(ymin + dwidth))
    img[xmin:xmax,ymin:ymax,:] = np.random.random_integers(0, 256, (xmax-xmin, ymax-ymin, 3))
    return img


def image_enforcing(img, flag=0, contrast=(0.5, 2.5), bright=(-50, 50), rotation=(-15, 15)):
    if flag == 1:  # trans hue
        #img = cv2.convertScaleAbs(img, alpha=random.uniform(*contrast), beta=random.uniform(*bright))
        pass
    elif flag == 2:  # rotation
        height, width = img.shape[:-1]
        matRotate = cv2.getRotationMatrix2D((height, width), random.randint(-15, 15), 1) # mat rotate 1 center 2 angle 3 缩放系数
        img = cv2.warpAffine(img, matRotate, (height, width))
        pass
    elif flag == 3:  # flp 翻转
        img = cv2.flip(img, 1)
    return img


def two_point(age_label, category, interval=10, elips=0.000001):
    def age_split(age):
        embed = [0 for x in range(0, category)]
        right_prob = age % interval * 1.0 / interval
        left_prob = 1 - right_prob
        idx = age // interval
        if left_prob:
            embed[idx] = left_prob
        if right_prob and idx + 1 < category:
            embed[idx+1] = right_prob
        return embed
    return np.array(age_split(age_label))



def data_generator(dataframe, batch_size=32, category=12, interval=10, is_training=True, dropout=0.):
    dataframe = dataframe.reset_index(drop=True)
    all_nums = len(dataframe) 
    while True:
        idxs = np.random.permutation(all_nums)
        start = 0
        while start + batch_size < all_nums:
            candis = dataframe.loc[list(idxs[start:start+batch_size])]
            # process on batch
            imgs_x = []
            agev_y = []
            #row_iter = candis.iterrows()
            for i in range(len(candis)):
                row = candis.iloc[i] #next(row_iter)
                # print('row_type', type(row))
                # Age vector and scalar
                agev_y.append([row['age'], two_point(row['age'], category, interval)])
                # print('row__', row)
                # Image Transform
                imgs_x.append(image_transform(row))
            imgs_x = np.array(imgs_x)        
            yield [imgs_x[:,0], imgs_x[:,1], imgs_x[:,2]], agev_y
            start += batch_size