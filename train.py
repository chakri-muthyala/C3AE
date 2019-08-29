#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:39:18 2019

@author: chakri
"""

#import tensorflow as tf
#import keras
#from keras.layers import BatchNormalization, Conv2D, ReLU, GlobalAveragePooling2D, multiply, GlobalMaxPooling2D, AveragePooling2D, Concatenate
#from keras.models import Model, load_model
#from keras.engine.input_layer import Input
#from keras.layers.core import Dense, Dropout
#from keras.activations import softmax, sigmoid
#from keras import regularizers
#from keras.layers import Lambda, Multiply, multiply


#coding=utf-8
import cv2
import os
import re
#import feather
import base64
import math
import numpy as npConcatenate
import pandas as pd
import random
import tensorflow as tf
import keras.backend as K
import keras
from keras.optimizers import Adam
from keras.activations import softmax, sigmoid
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Model, load_model
from keras.engine.input_layer import Input
from keras.layers import BatchNormalization, Conv2D, ReLU, GlobalAveragePooling2D, multiply, GlobalMaxPooling2D, AveragePooling2D, Concatenate
from keras.layers.core import Dense, Dropout 
from keras.layers import Lambda, Multiply, multiply
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, LambdaCallback
from sklearn.model_selection import train_test_split
from keras.backend import argmax, pool2d
from keras import backend as K
from keras import regularizers
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


#custom
import xy_generator
import utils




# c3ae Netimport tensorflow as tf
#import keras
#from keras.layers import BatchNormalization, Conv2D, ReLU, GlobalAveragePooling2D, multiply, GlobalMaxPooling2D, AveragePooling2D
#from keras.models import Model, load_model
#from keras.engine.input_layer import Input
#from keras.layers.core import Dense, Dropout
#from keras.activations import softmax, sigmoid
#from keras import regularizers
#from keras.layers import Lambda, Multiply, multiply


def plain_model(height=64, width=64, channel=3):
    #c3ae naive net
    input_image = Input(shape=[64, 64, 3])
    #conv1
    conv_1 = Conv2D(32, (3, 3), padding="valid", strides=1, use_bias=False)(input_image)
    #bravg1
    bn_1 = BatchNormalization()(conv_1)
    rl_1 = ReLU()(bn_1) #have to be implemented with relu6
    ap2_1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(rl_1)
    #conv2
    conv_2 = Conv2D(32, (3, 3), padding="valid", strides=1)(ap2_1)
    #bravg2
    bn_2 = BatchNormalization()(conv_2)
    rl_2 = ReLU()(bn_2) #have to be implemented with relu6
    ap2_2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(rl_2)
    #conv3
    conv_3 = Conv2D(32, (3, 3), padding="valid", strides=1)(ap2_2)
    #bravg3
    bn_3 = BatchNormalization()(conv_3)
    rl_3 = ReLU()(bn_3) #have to be implemented with relu6
    ap2_3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(rl_3)
    #conv4
    conv_4 = Conv2D(32, (3, 3), padding="valid", strides=1)(ap2_3)
    #BN+ReLu
    bn_4 = BatchNormalization()(conv_4)
    rl_4 = ReLU()(bn_4) #have to be implemented with relu6
    #conv5
    conv_5 = Conv2D(32, (1, 1), padding="valid", strides=1)(rl_4)
    
    feat_conv1 = Conv2D(12, (4, 4), padding="valid", strides=1)(conv_5)
    #feat_conv2 = Conv2D(12, (1, 1), padding="valid", strides=4)(feat_conv1)
    
    featw1 = GlobalAveragePooling2D()(feat_conv1)
    feats = Dense(12, activation=softmax, kernel_regularizer=regularizers.l1(0), name="W1")(featw1)
    pmodel = Model(input=input_image, output=feats)
    
    return pmodel


def shared_model(CATES=10, height=64, width=64, channel=3):
    
    
    base_model = plain_model()
    print(base_model.summary())
    x1 = Input(shape=(height, width, channel))
    x2 = Input(shape=(height, width, channel))
    x3 = Input(shape=(height, width, channel))

    y1 = base_model(x1)
    y2 = base_model(x2)
    y3 = base_model(x3)

    cfeat = Concatenate()([y1, y2, y3])
    bulk_feat = Dense(CATES, use_bias=True, activity_regularizer=regularizers.l1(0), activation=softmax)(cfeat)
    age = Dense(1, name="age")(bulk_feat)
    #age = Lambda(lambda a: tf.reshape(tf.reduce_sum(a * tf.constant([[x * 10.0 for x in xrange(12)]]), axis=-1), shape=(-1, 1)), name="age")(bulk_feat)
    return Model(input=[x1, x2, x3], output=[age, bulk_feat])


def config_gpu():
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    
    
def preprocessing(dataframes, batch_size=50, category=10, interval=10, is_training=True, dropout=0.):
    # category: bin + 2 due to two side
    # interval: age interval 
    from xy_generator import data_generator
    return data_generator(dataframes, category=category, interval=interval, batch_size=batch_size, is_training=is_training, dropout=dropout)



def train():
    config_gpu()

    sample_rate = 0.8
    seed = 2019
    batch_size = 32
    category = 10
    interval = int(math.ceil(100. / category))
    lr = 0.002

    dataframes = utils.load_pickle(file_path = 'pickle_1.dat')

    trainset, testset = train_test_split(dataframes, train_size=sample_rate, test_size=1-sample_rate, random_state=seed)

    dropout = 0.2

    train_gen = preprocessing(trainset, dropout=dropout, category=category, interval=interval)
    validation_gen = preprocessing(testset, is_training=False, category=category, interval=interval)
    #print(trainset.groupby(["age"]).agg(["count"]))
    #print(testset.groupby(["age"]).agg(["build_shareable_cnncount"]))

    save_path = "./model/c3ae_model_v2.h5"
    log_dir = ''
    weight_factor = 10

    model = shared_model()
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0001, amsgrad=False)
    model.compile(
        optimizer=adam,
        loss=["mean_absolute_error", "kullback_leibler_divergence"],
        metrics={"age": "mae", "W1": "mae"},
        loss_weights=[1, weight_factor]
    )
    W2 = model.get_layer("age")

    def get_weights(epoch, loggs):
        print('')#epoch, K.get_value(model.optimizer.lr), W2.get_weights())
    callbacks = [
        ModelCheckpoint(save_path, monitor='val_age_mean_absolute_error', verbose=1, save_best_only=True, mode='min'),
        ModelCheckpoint("train_" + save_path, monitor='age_mean_absolute_error', verbose=1, save_best_only=True, mode='min'),
        TensorBoard(log_dir=log_dir, batch_size=batch_size, write_images=True, update_freq='epoch'),
        #EarlyStopping(monitor='val_age_mean_absolute_error', patience=10, verbose=0, mode='min'),
        #LearningRateScheduler(lambda epoch: lr - 0.0001 * epoch // 10),
        ReduceLROnPlateau(monitor='age_mean_absolute_error', factor=0.2, patience=10, min_lr=0.00001),
        LambdaCallback(on_epoch_end=get_weights)
        ]
    history = model.fit_generator(train_gen, steps_per_epoch=len(trainset) / batch_size, epochs=600, callbacks=callbacks, validation_data=validation_gen, validation_steps=len(testset) / batch_size * 3)
    
#    fgen = image_generator.data_generator(dataframes, category=10, interval=10, batch_size=50, is_training=True, dropout=0.)
#    x = []
#    y = []
#    for img, ags in fgen:
#        x.append(img)
#        y.append(ags)
#
#    history = model.fit(x=x, y=y, epochs=600, callbacks=callbacks, validation_split=1-sample_rate, steps_per_epoch=300, validation_steps=300)

    
if __name__ == '__main__':
    train()
    
    