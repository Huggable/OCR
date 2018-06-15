import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import tensorflow.contrib.slim as slim
import time
import logging
import numpy as np
import tensorflow as tf
import pickle
import shutil
from PIL import Image
import cv2
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib.slim.python.slim.nets import resnet_v2 as res
from tensorflow.contrib.slim.python.slim.nets import inception_v3 as inc
sys.path.append('./')

x = tf.Variable(0.0)
x_plus_1 = tf.assign_add(x, 1)
y = x
init = tf.global_variables_initializer()
with tf.control_dependencies([x_plus_1]):
    y = tf.identity(x)
with tf.Session() as session:
    init.run()
    for i in range(5):
        print(y.eval())
