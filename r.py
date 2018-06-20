import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import sys
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
import gen_printed_char as g
import own as o


INS_SIZE = 299
RES_SIZE = 224



logger = logging.getLogger('Training a chinese write char recognition')
logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


tf.app.flags.DEFINE_string('graph', 'OCR', 'the graph u wanna run')
tf.app.flags.DEFINE_string('optimizer', 'A', 'the optimizer u wanna run')
tf.app.flags.DEFINE_string('learning_rate', '0.01', 'the optimizer u wanna run')
tf.app.flags.DEFINE_integer('max_steps', 16003, 'the max training steps ')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Validation batch size')
tf.app.flags.DEFINE_integer('eval_steps', 10, "the step num to eval")
tf.app.flags.DEFINE_integer('save_steps', 4000, "the steps to save")
tf.app.flags.DEFINE_boolean('restore', False, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_integer('charset_size', 5210, "Choose the first `charset_size` characters only.")
tf.app.flags.DEFINE_string('dictionarydir', './dictionary.txt', 'dictioary dir"}')
tf.app.flags.DEFINE_string('mode', 'train', 'Running mode. One of {"train", "valid", "inference"}')
tf.app.flags.DEFINE_string('inferencedir', './tmp', 'the graph u wanna run')
Gradient_dict = {'G': tf.train.GradientDescentOptimizer,
                 'A': tf.train.AdamOptimizer,
                 'R': tf.train.RMSPropOptimizer}





tf.app.flags.DEFINE_boolean('random_flip_up_down', False, "Whether to random flip up down")
tf.app.flags.DEFINE_boolean('random_brightness', True, "whether to adjust brightness")
tf.app.flags.DEFINE_boolean('random_contrast', True, "whether to random constrast")


tf.app.flags.DEFINE_boolean('gray', True, "whether to change the rbg to gray")


tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('train_data_dir', './dataset/train/', 'the train dataset dir')
tf.app.flags.DEFINE_string('test_data_dir', './dataset/test/', 'the test dataset dir')
tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')


tf.app.flags.DEFINE_integer('epoch', 1, 'Number of epoches')


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
FLAGS = tf.app.flags.FLAGS



class DataIterator:
    def __init__(self, data_dir, size):
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
        self.s = size

        f = open(FLAGS.dictionarydir, 'r')
        a = f.read()
        lang_char = eval(a)
        f.close()
        '''
        dict_new = dict(zip(lang_char.values(), lang_char.keys()))
        for i in range(len(self.image_names)):
            print(self.image_names[i],dict_new[self.labels[i]])
        '''

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
        new_size = tf.constant([self.s, self.s], dtype=tf.int32)
        images = tf.image.resize_images(images, new_size)
        image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=50000,
                                                          min_after_dequeue=10000)
        # print 'image_batch', image_batch.get_shape()
        return image_batch, label_batch

def build_graph(top_k=1,op = FLAGS.optimizer):
    keep_prob = tf.constant(value=0.8,dtype=tf.float32, shape=[], name='keep_prob') # dropout打开概率
    images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name='image_batch')
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
    is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')
    learn_rate = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')


    # network: conv2d->max_pool2d->conv2d->max_pool2d->conv2d->max_pool2d->conv2d->conv2d->
    # max_pool2d->fully_connected->fully_connected
    #给slim.conv2d和slim.fully_connected准备了默认参数：batch_norm
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        normalizer_fn=tf.layers.batch_normalization,
                        normalizer_params={'training': is_training}):
        conv3_1 = slim.conv2d(images, 64, [3, 3], 1, padding='SAME', scope='conv3_1')
        max_pool_1 = slim.max_pool2d(conv3_1, [2, 2], [2, 2], padding='SAME', scope='pool1')
        conv3_2 = slim.conv2d(max_pool_1, 128, [3, 3], padding='SAME', scope='conv3_2')
        max_pool_2 = slim.max_pool2d(conv3_2, [2, 2], [2, 2], padding='SAME', scope='pool2')
        conv3_3 = slim.conv2d(max_pool_2, 256, [3, 3], padding='SAME', scope='conv3_3')
        max_pool_3 = slim.max_pool2d(conv3_3, [2, 2], [2, 2], padding='SAME', scope='pool3')
        conv3_4 = slim.conv2d(max_pool_3, 512, [3, 3], padding='SAME', scope='conv3_4')
        conv3_5 = slim.conv2d(conv3_4, 512, [3, 3], padding='SAME', scope='conv3_5')
        max_pool_4 = slim.max_pool2d(conv3_5, [2, 2], [2, 2], padding='SAME', scope='pool4')

        flatten = slim.flatten(max_pool_4)
        fc1 = slim.fully_connected(slim.dropout(flatten, keep_prob), 1024,
                                   activation_fn=tf.nn.relu, scope='fc1')
        logits = slim.fully_connected(slim.dropout(fc1, keep_prob), FLAGS.charset_size, activation_fn=None,
                                      scope='fc2')
    # 因为我们没有做热编码，所以使用sparse_softmax_cross_entropy_with_logits

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))


    global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    optimizer = Gradient_dict[op](learning_rate=learn_rate)

    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss , global_step=global_step)


    #train_op = slim.learning.create_train_op(loss, optimizer, global_step=global_step)
    probabilities = tf.nn.softmax(logits)
    predicition = tf.argmax(probabilities,1)

    # 绘制loss accuracy曲线
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    # 返回top k 个预测结果及其概率；返回top K accuracy
    predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
    accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))

    return {'images': images,
            'labels': labels,
            'keep_prob': keep_prob,
            'top_k': top_k,
            'global_step': global_step,
            'train_step': train_op,
            'loss': loss,
            'is_training': is_training,
            'accuracy': accuracy,
            'accuracy_top_k': accuracy_in_top_k,
            'merged_summary_op': merged_summary_op,
            'probabilities': probabilities,
            'predicted_index_top_k': predicted_index_top_k,
            'predicted_val_top_k': predicted_val_top_k,
            'learning_rate':learn_rate,
            'predicition':predicition}

def build_graph_inc(op = FLAGS.optimizer):
    images = tf.placeholder(dtype=tf.float32, shape=[None, INS_SIZE, INS_SIZE, 1], name='image_batch')
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
    learn_rate = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
    is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')

    net, end_points = inc.inception_v3(inputs = images, num_classes = FLAGS.charset_size, is_training = is_training)
    net = tf.reshape(net,[-1, FLAGS.charset_size])



    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net, labels=labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = Gradient_dict[op](learn_rate).minimize(cross_entropy_mean)

    probabilities = tf.nn.softmax(net)
    predicition = tf.argmax(probabilities,1)

    reout = tf.reshape(end_points['Predictions'],[-1,FLAGS.charset_size])


    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(reout, 1), labels)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('loss', cross_entropy_mean)
    tf.summary.scalar('accuracy', evaluation_step)
    merged_summary_op = tf.summary.merge_all()

    return {'images':images,
            'labels':labels,
            'loss':cross_entropy_mean,
            'accuracy':evaluation_step,
            'train_step':train_step,
            'merged_summary_op':merged_summary_op,
            'learning_rate':learn_rate,
            'is_training':is_training,
            'predicition':predicition}


def build_graph_res(op = FLAGS.optimizer):
    images = tf.placeholder(dtype=tf.float32, shape=[None, RES_SIZE, RES_SIZE, 1], name='image_batch')
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
    learn_rate = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
    is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')


    net, end_points = res.resnet_v2_50(inputs = images, num_classes = FLAGS.charset_size, is_training = is_training)
    net = tf.reshape(net,[-1, FLAGS.charset_size])



    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net, labels=labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = Gradient_dict[op](learn_rate).minimize(cross_entropy_mean)


    probabilities = tf.nn.softmax(net)
    predicition = tf.argmax(probabilities,1)


    reout = tf.reshape(end_points['predictions'],[-1,FLAGS.charset_size])


    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(reout, 1), labels)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('loss', cross_entropy_mean)
    tf.summary.scalar('accuracy', evaluation_step)
    merged_summary_op = tf.summary.merge_all()

    return {'images':images,
            'labels':labels,
            'loss':cross_entropy_mean,
            'accuracy':evaluation_step,
            'train_step':train_step,
            'merged_summary_op':merged_summary_op,
            'learning_rate':learn_rate,
            'is_training':is_training,
            'predicition':predicition}



def test(op = FLAGS.optimizer):
    images = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 1], name='image_batch')
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
    learn_rate = tf.placeholder(dtype = tf.float32, shape=[], name = 'learning_rate')
    is_training = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        normalizer_fn=tf.layers.batch_normalization,
                        normalizer_params={'training': is_training}):
        cov1 = slim.conv2d(images, 64, [3,3],scope= 'cov1')
        max1 = slim.max_pool2d(cov1, [2, 2], [2, 2], padding='SAME', scope='pool1')
        cov2 = slim.conv2d(max1, 128, [3, 3], scope='cov2')
        max2 = slim.max_pool2d(cov2, [2, 2], [2, 2], padding='SAME', scope='pool2')
        cov3 = slim.conv2d(max2, 256, [3, 3], scope='cov3')
        max3 = slim.max_pool2d(cov3, [2, 2], [2, 2], padding='SAME', scope='pool3')
        cov4 = slim.conv2d(max3, 512, [3, 3], scope='cov4')
        cov5 = slim.conv2d(cov4, 1024, [3, 3], scope='cov5')
        cov6 = slim.conv2d(cov5, 1024, [3, 3], scope='cov6')
        flatten = slim.flatten(cov6)
        fc1 = slim.fully_connected(flatten, 1024,activation_fn=tf.nn.relu, scope='fc1')
        fc2 = slim.fully_connected(fc1, FLAGS.charset_size, activation_fn=None, scope='fc2')


        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc2, labels=labels))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fc2, 1), labels), tf.float32))
        global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        optimizer = Gradient_dict[op](learning_rate=learn_rate)
        gv = optimizer.compute_gradients(loss)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(gv,global_step = global_step)


        for g, v in gv:
            if g is not None:
                if 'weights' in v.name:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))






        probabilities = tf.nn.softmax(fc2)

        predicition = tf.argmax(probabilities, 1)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        merged_summary_op = tf.summary.merge_all()
        return {'images': images,
                'labels': labels,
                'loss': loss,
                'accuracy': accuracy,
                'train_step': train_op,
                'merged_summary_op': merged_summary_op,
                'learning_rate':learn_rate,
                'is_training':is_training,
                'probabilities':probabilities,
                'predicition':predicition,
                'global_step': global_step}



def train():
    graph_dict = {'res': [build_graph_res, RES_SIZE],
                  'inc': [build_graph_inc, INS_SIZE],
                  'OCR' : [build_graph, 64],
                  'testing' : [test, 128],
                  'newcnn' : [o.new_cnn, 64]}
    graph = graph_dict[FLAGS.graph][0]()

    train_feeder = DataIterator(data_dir='./dataset/train/', size = graph_dict[FLAGS.graph][1])
    test_feeder1 = DataIterator(data_dir='./dataset/train/', size = graph_dict[FLAGS.graph][1])
    test_feeder2 = DataIterator(data_dir='./dataset/test/', size=graph_dict[FLAGS.graph][1])
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:

        train_images, train_labels = train_feeder.input_pipeline(batch_size=FLAGS.batch_size, aug=True)
        test_images1, test_labels1 = test_feeder1.input_pipeline(batch_size=FLAGS.batch_size)
        test_images2, test_labels2 = test_feeder2.input_pipeline(batch_size=FLAGS.batch_size)

        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars

        #saver = tf.train.Saver(var_list=var_list, max_to_keep = 1)
        saver = tf.train.Saver(max_to_keep = 5)
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)




        summary_dir = FLAGS.log_dir + '/train/' + FLAGS.graph + '_' + FLAGS.optimizer + '_' + str(FLAGS.learning_rate)
        if FLAGS.restore == False:
            try:
                shutil.rmtree(summary_dir)
            except Exception:
                pass
            finally:
                os.mkdir(summary_dir)
        train_writer = tf.summary.FileWriter(summary_dir, sess.graph)



        summary_dir_test = FLAGS.log_dir + '/val/' + FLAGS.graph + '_' + FLAGS.optimizer + '_' + str(FLAGS.learning_rate)+'train'
        if FLAGS.restore == False:
            try:
                shutil.rmtree(summary_dir_test)
            except Exception:
                pass
            finally:
                os.mkdir(summary_dir_test)
        test_writer1 = tf.summary.FileWriter(summary_dir_test)

        summary_dir_test = FLAGS.log_dir + '/val/' + FLAGS.graph + '_' + FLAGS.optimizer + '_' + str(FLAGS.learning_rate)+'test'
        if FLAGS.restore == False:
            try:
                shutil.rmtree(summary_dir_test)
            except Exception:
                pass
            finally:
                os.mkdir(summary_dir_test)
        test_writer2 = tf.summary.FileWriter(summary_dir_test)








        start_step = 0
        # 可以从某个step下的模型继续训练
        ckpt_dir = FLAGS.checkpoint_dir + FLAGS.graph + '_' + FLAGS.optimizer + '_' + str(FLAGS.learning_rate)
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(ckpt_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                print("restore from the checkpoint {0}".format(ckpt))
                start_step += int(ckpt.split('-')[-1])

        logger.info(':::Training Start:::')
        i = start_step



        try:
            while not coord.should_stop():
                i += 1
                if i > FLAGS.max_steps:
                    break
                start_time = time.time()
                train_images_batch, train_labels_batch = sess.run([train_images, train_labels])
                feed_dict = {graph['images']: train_images_batch,
                            graph['labels']: train_labels_batch,
                            graph['learning_rate']: float(FLAGS.learning_rate),
                            graph['is_training']: True}

                _, loss, summary, step  = sess.run([graph['train_step'],
                                    graph['loss'],
                                    graph['merged_summary_op'],
                                    graph['global_step']],
                                    feed_dict=feed_dict)

                train_writer.add_summary(summary, step)
                end_time = time.time()
                logger.info(
                    "the step {0} takes {1} loss {2} ".format(i, end_time - start_time, loss))
                if step % FLAGS.save_steps == 0:
                    logger.info('Save the ckpt of {0}'.format(i))
                    if not os.path.exists(ckpt_dir):
                        os.mkdir(ckpt_dir)
                    saver.save(sess, os.path.join(ckpt_dir, 'C'), global_step = graph['global_step'])
                if step % FLAGS.eval_steps == 0:
                    test_images_batch, test_labels_batch = sess.run([test_images1, test_labels1])

                    feed_dict = {graph['images']: test_images_batch,
                                 graph['labels']: test_labels_batch,
                                 graph['is_training']: False}
                    accuracy, summary = sess.run([graph['accuracy'],
                                            graph['merged_summary_op']],
                                            feed_dict = feed_dict)
                    test_writer1.add_summary(summary, step)
                    test_images_batch, test_labels_batch = sess.run([test_images2, test_labels2])

                    feed_dict = {graph['images']: test_images_batch,
                                 graph['labels']: test_labels_batch,
                                 graph['is_training']: False}
                    accuracy, summary = sess.run([graph['accuracy'],
                                                  graph['merged_summary_op']],
                                                 feed_dict=feed_dict)
                    test_writer2.add_summary(summary, step)
                    logger.info("---------val accuracy {0} ".format(accuracy))
        except tf.errors.OutOfRangeError:
            logger.info('==================Train Finished================')
        finally:
           pass

def validation():
    print('Begin validation')

    graph_dict = {'res': [build_graph_res, RES_SIZE],
                  'inc': [build_graph_inc, INS_SIZE],
                  'OCR' : [build_graph, 64],
                  'testing' : [test, 128],
                  'newcnn' : [o.new_cnn, 64]}

    graph = graph_dict[FLAGS.graph][0]()
    test_feeder = DataIterator(data_dir='./dataset/test/', size=graph_dict[FLAGS.graph][1])
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
        test_images, test_labels = test_feeder.input_pipeline(batch_size=FLAGS.batch_size, num_epochs=1)

        saver = tf.train.Saver(max_to_keep = 5)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)



        summary_dir_test = FLAGS.log_dir + '/val/' + FLAGS.graph + '_' + FLAGS.optimizer + '_' + str(FLAGS.learning_rate)
        if FLAGS.restore == False:
            try:
                shutil.rmtree(summary_dir_test)
            except Exception:
                pass
            finally:
                os.mkdir(summary_dir_test)
        test_writer = tf.summary.FileWriter(summary_dir_test)


        ckpt_dir = FLAGS.checkpoint_dir + FLAGS.graph + '_' + FLAGS.optimizer + '_' + str(FLAGS.learning_rate)
        ckpt = tf.train.latest_checkpoint(ckpt_dir)
        if ckpt:
            print(ckpt)
            print(type(ckpt))
            #ckpt = './checkpoint/testing_A_0.001/C-10000'
            saver.restore(sess, ckpt)
            print("restore from the checkpoint {0}".format(ckpt))

        try:
            i = 0
            while not coord.should_stop():
                i += 1
                if i > 10000:
                    break
                start_time = time.time()
                test_images_batch, test_labels_batch = sess.run([test_images, test_labels])
                feed_dict = {graph['images']: test_images_batch,
                            graph['labels']: test_labels_batch,
                            graph['learning_rate']: float(FLAGS.learning_rate),
                            graph['is_training']: False}

                accuracy , loss, summary  = sess.run([graph['accuracy'],
                                    graph['loss'],
                                    graph['merged_summary_op']],
                                    feed_dict=feed_dict)

                test_writer.add_summary(summary , i)
                end_time =time.time()
                logger.info(
                    "the step {0} takes {1} loss {2} accuracy {3} ".format(i, end_time - start_time, loss, accuracy))
        except Exception:
            pass
        finally:
            pass








def get_file_list(path):
    list_name=[]
    files = os.listdir(path)
    files.sort()
    for file in files:
        file_path = os.path.join(path, file)
        list_name.append(file_path)
    return list_name


def binary_pic(name_list):
    i = 0
    dir = './data/temp'
    list = []
    try:
        shutil.rmtree(dir)
    except Exception:
        pass
    finally:
        os.mkdir(dir)
    for image in name_list:
        temp_image = cv2.imread(image)
        GrayImage=cv2.cvtColor(temp_image,cv2.COLOR_BGR2GRAY)
        ret,thresh1=cv2.threshold(GrayImage,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        newdir = dir +'/'+ ("%0.5d.png" % i)
        cv2.imwrite(newdir, thresh1)
        list.append(newdir)
        i += 1
    return list

def generate_test():

    g.white=True;
    font_dir = './testfont'
    width = 10
    height = 10

    output = '放假了终于可以继续可以静下心写一写OCR方面的东西上次谈到文字的切割今天打算总结一下我们怎么得到用于训练的文字数据集如果是想训练一个手写体识别的模型用一些前人收集好的手写文字集就好了比如中科院的这些数据集'

    f = open('./dictionary.txt', 'r')
    a = f.read()
    lang_char = eval(a)
    f.close()
    dict_new = dict(zip(lang_char.values(), lang_char.keys()))
    outputlist1 = []
    for char in output:
        outputlist1.append(lang_char[char])


    font_check = g.FontCheck(lang_char)
    verified_font_paths = []
    # search for file fonts
    for font_name in os.listdir(font_dir):
        path_font_file = os.path.join(font_dir, font_name)
        if font_check.do(path_font_file):
            verified_font_paths.append(path_font_file)
            print(path_font_file)

    font2image = g.Font2Image(width, height, True, 4)

    outputlist = []
    image_list = []
    for j, verified_font_path in enumerate(verified_font_paths):
        for char in output:
            image = font2image.do(verified_font_path, char)
            image_list.append(image)
        outputlist.extend(outputlist1)

    count = 0
    char_dir = './tmp'
    try:
        shutil.rmtree(char_dir)
    except Exception:
        pass
    finally:
        os.mkdir(char_dir)
    print(len(image_list))
    for i in range(len(image_list)):
        img = image_list[i]
        # print(img.shape)
        path_image = os.path.join(char_dir, "%0.5d.png" % count)
        cv2.imwrite(path_image, img)
        count += 1

    charlist = []
    for i in outputlist:
        charlist.append(dict_new[i])
    print(charlist)
    return  outputlist


def inference():
    print('inference')
    outputlist = generate_test()
    graph_dict = {'res': [build_graph_res, RES_SIZE],
                  'inc': [build_graph_inc, INS_SIZE],
                  'OCR' : [build_graph, 64],
                  'testing' : [test, 128],
                  'newcnn': [o.new_cnn, 64]}
    graph = graph_dict[FLAGS.graph][0]()


    f = open(FLAGS.dictionarydir, 'r')
    a = f.read()
    lang_char = eval(a)
    f.close()


    name_list = get_file_list(FLAGS.inferencedir)

    #二值化
    name_list = binary_pic(name_list)
    print(name_list)


    dict_new = dict(zip(lang_char.values(), lang_char.keys()))






    image_set = []
    # 对每张图进行尺寸标准化和归一化
    for image in name_list:
        temp_image = Image.open(image).convert('L')
        temp_image = temp_image.resize((graph_dict[FLAGS.graph][1], graph_dict[FLAGS.graph][1]), Image.ANTIALIAS)
        temp_image = np.asarray(temp_image) / 255.0
        temp_image = temp_image.reshape([-1, graph_dict[FLAGS.graph][1], graph_dict[FLAGS.graph][1], 1])
        image_set.append(temp_image)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True)) as sess:
        logger.info('========start inference============')
        # images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
        # Pass a shadow label 0. This label will not affect the computation graph.




        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        ckpt_dir = FLAGS.checkpoint_dir + FLAGS.graph + '_' + FLAGS.optimizer + '_' + str(FLAGS.learning_rate)
        ckpt = tf.train.latest_checkpoint(ckpt_dir)
        if ckpt:
            print("-----------------------------------------------------------")
            saver.restore(sess, ckpt)
            print("restore from the checkpoint {0}".format(ckpt))

        val_list = []

        p = 0
        for i in range(len(image_set)):
            temp_image = image_set[i]
            predicition = sess.run([graph['predicition']],
                                    feed_dict={graph['images']: temp_image,
                                                graph['is_training']: False})
            val_list.append(dict_new[int(predicition[0][0])])
            if not int(predicition[0][0]) == outputlist[i]:
                p += 1
                print('should be ',dict_new[outputlist[i]],'      now is ', dict_new[int(predicition[0][0])])


        print(val_list)
        print('    Accuracy:',1-(p+1)/(i+1))


def main(_):
    if FLAGS.mode == 'train':
        train()
    if FLAGS.mode == 'inference':
        inference()
    if FLAGS.mode == 'validation':
        validation()


if __name__ == "__main__":
    tf.app.run()
    #test()