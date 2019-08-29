#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 15:51:11 2019

@author: chakri
"""



import pickle
import pandas as pd


def load_pickle(file_path,):
    with open(file_path, 'rb') as f:
        dataframe = pickle.load(f)
    dataframe = pd.DataFrame(dataframe)
    dataframe = dataframe[(dataframe["age"] > 0) & (dataframe["age"] < 101)]
    dataframe = dataframe[(dataframe["yaw"] >-30) & (dataframe["yaw"] < 30) & (dataframe["roll"] >-20) & (dataframe["roll"] < 20) & (dataframe["pitch"] >-20) & (dataframe["pitch"] < 20) ]

    return process_unbalance(dataframe)

def process_unbalance(dataframe, max_nums=500, random_seed=2019):
    sample = []
    for x in range(100): # Age(0,100)
        age_set = dataframe[dataframe.age == x]
        cur_age_num = len(age_set)
        if cur_age_num > max_nums:
            age_set = age_set.sample(max_nums, random_state=random_seed, replace=False)
        sample.append(age_set)
    return pd.concat(sample, ignore_index=True)