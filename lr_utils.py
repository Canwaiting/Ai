import numpy as np
import h5py
    
    

def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r") #上同，把h5py格式的文件以read形式读入
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    #  保存的是以bytes类型保存的两个字符串数据，数据为：[b’non-cat’ b’cat’]
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))   #不知道train_set_y_orig.shape[0]是用来干什么的
    #.shape[0]表示输出行数[1]表示输出列数，如果里面的换成k，则输出的是行数列数
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

