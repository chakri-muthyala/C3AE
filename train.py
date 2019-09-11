#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 09:45:22 2019

@author: chakri
"""

import math

import keras
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, LambdaCallback
from sklearn.model_selection import train_test_split
import keras.backend as K

from model import shared_plain_model
from utils import load_pickle, data_generator


def config_gpu():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

if __name__ == '__main__':
    config_gpu()
    sample_rate = 0.8
    seed = 2019
    batch_size = 32
    category = 12
    interval = int(math.ceil(100. / category))
    lr = 0.002
    dropout=0.2
    
    dataframe = load_pickle('MORPH.dat')
    trainset, testset = train_test_split(dataframe, train_size=sample_rate, test_size=1-sample_rate, random_state=seed)
    train_gen = data_generator(trainset, dropout=dropout, category=category, interval=interval)
    validation_gen = data_generator(testset, is_training=False, category=category, interval=interval)
    
    model = shared_plain_model()
    
    save_path = "./model/c3ae_model_v2.h5"
    log_dir = ''
    weight_factor = 1
    
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0001, amsgrad=False)
    
    model.compile(
        optimizer=adam,
        loss=["mean_absolute_error", "kullback_leibler_divergence"],
        metrics={"age": "mae", "W1": "mae"},
        loss_weights=[1, weight_factor]
    )
    W2 = model.get_layer("age")
    
    def get_weights(epoch, loggs):
        print(epoch, K.get_value(model.optimizer.lr), W2.get_weights())

    callbacks = [
        ModelCheckpoint(save_path, monitor='val_age_mean_absolute_error', verbose=1, save_best_only=True, mode='min'),
        ModelCheckpoint(save_path, monitor='age_mean_absolute_error', verbose=1, save_best_only=True, mode='min'),
        TensorBoard(log_dir=log_dir, batch_size=batch_size, write_images=True, update_freq='epoch'),
        #EarlyStopping(monitor='val_age_mean_absolute_error', patience=10, verbose=0, mode='min'),
        #LearningRateScheduler(lambda epoch: lr - 0.0001 * epoch // 10),
        ReduceLROnPlateau(monitor='val_age_mean_absolute_error', factor=0.1, patience=10, min_lr=0.00001),
        LambdaCallback(on_epoch_end=get_weights)
    ]
    history = model.fit_generator(train_gen, steps_per_epoch=len(trainset) / batch_size, epochs=160, callbacks=callbacks, validation_data=validation_gen, validation_steps=len(testset) / batch_size * 3)

    