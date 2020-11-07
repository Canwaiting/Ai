#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import h5py


def load_dataset():
    train_dataset = h5py.File('dataset/train_catvnoncat.h5','r') #把数据读取
    train_set_x_orig = np.arry(train_dataset["train_set_x"][:])  #读取把数据集中的图像数据读取
    train_set_y_orig = np.arry(train_dataset["train_set_y"][:])  #读取把数据集中的判断值读取
    
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r") #与上面类似，只不过是测试集，上面的是训练集
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) #至于为什么要在test_dataset那里括号中增加字符串不懂
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) 

    classes = np.array(test_dataset["list_classes"][:]) #为什么要创建这个数组
    a = train_set_y_orig.shape[0]
    print(a)

