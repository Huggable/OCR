import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import tensorflow.contrib.slim as slim
import time
import logging
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib.slim.python.slim.nets import resnet_v2 as res
CLASS_NUM = 3755
lr = 0.0001

class DataIterator:
    def __init__(self, data_dir):
        # Set FLAGS.charset_size to a small value if available computation power is limited.
        truncate_path = data_dir + ('%05d' % FLAGS.charset_size)
        print(truncate_path)
        # 遍历训练集所有图像的路径，存储在image_names内
        self.image_names = []
        for root, sub_folder, file_list in os.walk(data_dir):
            if root < truncate_path:
                self.image_names += [os.path.join(root, file_path) for file_path in file_list]
        random.shuffle(self.image_names) # 打乱
        # 例如image_name为./train/00001/2.png，提取00001就是其label
        self.labels = [int(file_name[len(data_dir):].split(os.sep)[0]) for file_name in self.image_names]

    @property
    def size(self):
        return len(self.labels)

    @staticmethod
    def data_augmentation(images):
        # 镜像变换
        if FLAGS.random_flip_up_down:
            images = tf.image.random_flip_up_down(images)
        # 图像亮度变化
        if FLAGS.random_brightness:
            images = tf.image.random_brightness(images, max_delta=0.3)
        # 对比度变化
        if FLAGS.random_contrast:
            images = tf.image.random_contrast(images, 0.8, 1.2)
        return images
    # batch的生成
    def input_pipeline(self, batch_size, num_epochs=None, aug=False):
        # numpy array 转 tensor
        images_tensor = tf.convert_to_tensor(self.image_names, dtype=tf.string)
        labels_tensor = tf.convert_to_tensor(self.labels, dtype=tf.int64)
        # 将image_list ,label_list做一个slice处理
        input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor], num_epochs=num_epochs)

        labels = input_queue[1]
        images_content = tf.read_file(input_queue[0])
        images = tf.image.convert_image_dtype(tf.image.decode_png(images_content, channels=1), tf.float32)
        if aug:
            images = self.data_augmentation(images)
        new_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32)
        images = tf.image.resize_images(images, new_size)
        image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=50000,
                                                          min_after_dequeue=10000)
        # print 'image_batch', image_batch.get_shape()
        return image_batch, label_batch


def build_graph():
    images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name='image_batch')
    labels = tf.placeholder(dtype=tf.int64, shape=[None,CLASS_NUM], name='label_batch')
    global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)

    net,end_points = res.resnet_v2_50(inputs = images, num_classes = CLASS_NUM, is_training = True)


    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy_mean,global_step)


    reout = tf.reshape(end_points['predictions'],[-1,CLASS_NUM])



    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(reout, 1), tf.argmax(labels, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('loss', cross_entropy_mean)
    tf.summary.scalar('accuracy', evaluation_step)
    merged_summary_op = tf.summary.merge_all()
    print(evaluation_step.get_shape())
    return {'images':images,
            'labels':labels,
            'loss':cross_entropy_mean,
            'accuracy':evaluation_step,
            'train_step':train_step,
            'global_step':global_step,
            'merged_summary_op':merged_summary_op}

def train():

    with tf.Session() as sess:
        for i in range(100):
            graph = build_graph()
            feed_dict = {}
            _, step, loss, acc, summary=sess.run(graph['train'], graph['global_step'], graph['loss'], graph['accuracy'], graph['merged_summary_op'])
            log

build_graph()