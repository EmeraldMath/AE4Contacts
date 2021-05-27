#!/usr/bin/env python
# coding: utf-8

# In[31]:


# get_ipython().run_line_magic('load_ext', 'tensorboard')
import os
import pickle
import numpy as np
import config as conf
import csv
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
import sklearn.metrics
from PIL import Image
from io import BytesIO


# In[42]:


class contact_map(object):
    def image_no_diag(self, p):
        sz = tf.size(p)
        edge = tf.math.sqrt(tf.cast(sz, tf.float32))
        edge = tf.cast(edge, tf.int32)
        p = tf.reshape(p, [edge, edge])
        return p

    def image_diag(self, p):
        sz = tf.size(p)
        edge = tf.math.sqrt(tf.cast(sz, tf.float32))
        edge = tf.cast(edge, tf.int32)
        p = tf.reshape(p, [edge, edge])
        return p

    def decode_csv(self, line, decoy = False):
        line_split = tf.compat.v1.string_split([line], ',')
        feature = tf.strings.to_number(line_split.values[1:], tf.float32)
        label = line_split.values[0]
        label_split = tf.compat.v1.string_split([label], '/')
        name0 = label_split.values[-1]
        if decoy:
            name0 = label_split.values[-2] + '_' + label_split.values[-1]
        name_split = tf.compat.v1.string_split([name0], '.')
        name = name_split.values[0]
        p = self.image_no_diag(feature)
        return {"name": name, "contact": p, "diag": p}
    
    def __init__(self, ds_path = conf.DATASET_PATH, img_sz = conf.IMAGE_SIZE, max_ca = conf.MAX_CONTACT_AREA):
        self.ds_path = ds_path
        self.img_sz = img_sz
        self.max_ca = max_ca
        self.train = tf.data.TextLineDataset(ds_path + "ContactMap_train30.csv").map(lambda x: self.decode_csv(x))
        self.val_varseq = tf.data.TextLineDataset(ds_path + "ContactMap_val_varseq.csv").map(lambda x: self.decode_csv(x))
        self.robot = tf.data.TextLineDataset(ds_path + '3DRobot.csv').map(lambda x: self.decode_csv(x, True))
        self.it1 = tf.data.TextLineDataset(ds_path + 'I-TASSER-I.csv').map(lambda x: self.decode_csv(x))
        self.hit = tf.data.TextLineDataset(ds_path + '17H-I-TASSER-I.csv').map(lambda x: self.decode_csv(x))





