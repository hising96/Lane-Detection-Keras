# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 10:28:42 2018

@author: HIMANSHU
"""

kitti_train='C:/Users/HIMANSHU/Desktop/data_road/training/image_2'
import pickle
with open('kitti_train.p','wb') as f:
    pickle.dump(kitti_train,f)