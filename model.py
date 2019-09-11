#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 10:55:15 2019

@author: chakri
"""


from keras.engine.input_layer import Input
from keras.layers import BatchNormalization, Conv2D, ReLU, GlobalAveragePooling2D, multiply, GlobalMaxPooling2D, AveragePooling2D, Concatenate
from keras.layers.core import Dense
from keras.activations import softmax
from keras.models import Model
from keras import regularizers


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


def shared_plain_model(CATES=12, height=64, width=64, channel=3):
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