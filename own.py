
import tensorflow.contrib.slim as slim

import logging

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops


def add_cov_layer(input,inchannel,outchannel,kernel_shape,scope, is_training):
    with tf.variable_scope(scope):
        bias = tf.Variable(tf.random_normal([outchannel]), name='bias', dtype=tf.float32)
        weights = tf.Variable(tf.random_normal([kernel_shape, kernel_shape, inchannel, outchannel],mean=0.0, stddev=1.0), name='weights', dtype=tf.float32)
        conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME',name='cov_layer')
        bn = tf.layers.batch_normalization(conv+bias,training = is_training,name = 'bn')
        r = tf.nn.relu(bn , name='relu')
        tf.summary.histogram('weight', weights)
        tf.summary.histogram('bias' , bias)
    return r

def new_cnn():
    images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name='image_batch')
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
    learn_rate = tf.placeholder(dtype = tf.float32, shape=[], name = 'learning_rate')
    is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')

    cov1 = add_cov_layer(images, 1, 64, 3, scope='cov_1',is_training=is_training)
    max1 = tf.nn.max_pool(cov1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name='max1')
    cov2 = add_cov_layer(max1, 64, 128, 3, 'cov_2',is_training=is_training)
    max2 = tf.nn.max_pool(cov2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max2')
    cov3 = add_cov_layer(max2, 128, 256, 3, 'cov_3',is_training=is_training)
    max3 = tf.nn.max_pool(cov3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max3')
    cov4 = add_cov_layer(max3, 256, 512, 3, 'cov_4',is_training=is_training)
    max4 = tf.nn.max_pool(cov4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max4')
    flatten = slim.flatten(max4)
    fc1 = slim.fully_connected(slim.dropout(flatten, 0.5), 1024, activation_fn=tf.nn.relu, scope='fc1')
    fc = slim.fully_connected(slim.dropout(fc1, 0.5), 5210, activation_fn = None, scope = 'fc')


    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc, labels=labels))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fc, 1), labels), tf.float32))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)

    probabilities = tf.nn.softmax(fc)

    predicition = tf.argmax(probabilities, 1)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()

    return{'images': images,
            'labels': labels,
            'loss': loss,
            'accuracy': accuracy,
            'train_step': train_op,
            'merged_summary_op': merged_summary_op,
            'learning_rate':learn_rate,
            'is_training':is_training,
            'probabilities':probabilities,
            'predicition':predicition}



