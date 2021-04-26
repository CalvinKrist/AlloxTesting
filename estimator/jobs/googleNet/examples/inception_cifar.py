#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: inception_cifar.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import sys
import platform
import argparse
import numpy as np
import tensorflow as tf

sys.path.append('../')
import loader as loader
from src.nets.googlenet import GoogLeNet_cifar
from src.helper.trainer import Trainer
from src.helper.evaluator import Evaluator


PRETRINED_PATH = '/home/qge2/workspace/data/pretrain/inception/googlenet.npy'
IM_PATH = '../data/cifar/'

import importlib.util
spec = importlib.util.spec_from_file_location("time_writter", "../../../time_writter.py")
time_writter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(time_writter)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate the model')
    parser.add_argument('--predict', action='store_true',
                        help='Get prediction result')
    parser.add_argument('--finetune', action='store_true',
                        help='Fine tuning the model')
    parser.add_argument('--load', type=int, default=99,
                        help='Epoch id of pre-trained model')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--bsize', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--keep_prob', type=float, default=0.4,
                        help='Keep probability for dropout')
    parser.add_argument('--maxepoch', type=int, default=100,
                        help='Max number of epochs for training')

    parser.add_argument('--cifarPath', type=str, default="../../../data/cifar-10-batches-py",
                        help='Location of CIFAR dataset')
    parser.add_argument('--savePath', type=str, default=".",
                        help='Where to save model')

    parser.add_argument('--im_name', type=str, default='.png',
                        help='Part of image name')

    # Hardware flags
    parser.add_argument('--cpu', action='store_true',
                        help='To use CPU or not')
    parser.add_argument('--numThreads', type=int, default=16,
                        help='Number of CPU threads to use')
    parser.add_argument('--gpu', action='store_true',
                        help='To use GPU or not')
    parser.add_argument('--numGPUs', type=int, default=1,
                        help='The number of GPUs to use')

    return parser.parse_args()

def train():
    time_writter.Start()

    FLAGS = get_args()
    # Create Dataflow object for training and testing set
    train_data, valid_data = loader.load_cifar(
        cifar_path=FLAGS.cifarPath, batch_size=FLAGS.bsize, subtract_mean=True)

    pre_trained_path=None
    if FLAGS.finetune:
        # Load the pre-trained model (on ImageNet)
        # for convolutional layers if fine tuning
        pre_trained_path = PRETRINED_PATH

    # Create a training model
    train_model = GoogLeNet_cifar(
        n_channel=3, n_class=10, pre_trained_path=pre_trained_path,
        bn=True, wd=0, sub_imagenet_mean=False,
        conv_trainable=True, fc_trainable=True)
    train_model.create_train_model()
    # Create a validation model
    valid_model = GoogLeNet_cifar(
        n_channel=3, n_class=10, bn=True, sub_imagenet_mean=False)
    valid_model.create_test_model()

    # create a Trainer object for training control
    trainer = Trainer(train_model, valid_model, train_data, init_lr=FLAGS.lr)

    # Parse hardware
    config = None
    if FLAGS.cpu:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"
        config=tf.ConfigProto(inter_op_parallelism_threads=FLAGS.numThreads,
                   intra_op_parallelism_threads=FLAGS.numThreads,
                   device_count={'GPU':0, 'CPU':1})
    elif FLAGS.gpu:
        config=tf.ConfigProto(device_count={'GPU':FLAGS.numGPUs, 'CPU':1})
        visible_gpus = ''
        for gpu in range(FLAGS.numGPUs):
            visible_gpus += str(gpu) + ","
        config.gpu_options.visible_device_list=visible_gpus[:-1] # remove last comma
    else:
        raise Exception("Hardware not specified!")

    print("Starting experiment.")
    with tf.Session(config=config) as sess:
        #writer = tf.summary.FileWriter(FLAGS.savePath)
        #writer.add_graph(sess.graph)
        sess.run(tf.global_variables_initializer())
        for epoch_id in range(FLAGS.maxepoch):
            time_writter.LogUpdate()
            print("Epoch " + str(epoch_id) + " completed.")
            print(time_writter.GetResults())
            
            # train one epoch
            trainer.train_epoch(sess, keep_prob=FLAGS.keep_prob)

        time_writter.LogUpdate()
        print("Saving model...")
        saver = tf.train.Saver()
        saver.save(sess, '{}inception-cifar-epoch-{}'.format(FLAGS.savePath, epoch_id))
        #writer.close()

        print(time_writter.GetResults())
        print("Job completed.")


def evaluate():
    FLAGS = get_args()
    # Create Dataflow object for training and testing set
    train_data, valid_data = loader.load_cifar(
        cifar_path=FLAGS.cifarPath, batch_size=FLAGS.bsize, subtract_mean=True)
    # Create a validation model
    valid_model = GoogLeNet_cifar(
        n_channel=3, n_class=10, bn=True, sub_imagenet_mean=False)
    valid_model.create_test_model()

    # create a Evaluator object for evaluation
    evaluator = Evaluator(valid_model)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        # load pre-trained model cifar
        saver.restore(sess, '{}inception-cifar-epoch-{}'.format(FLAGS.savePath, FLAGS.load))
        print('training set:', end='')
        evaluator.accuracy(sess, train_data)
        print('testing set:', end='')
        evaluator.accuracy(sess, valid_data)

def predict():
    FLAGS = get_args()
    # Read Cifar label into a dictionary
    label_dict = loader.load_label_dict(dataset='cifar')
    # Create a Dataflow object for test images
    image_data = loader.read_image(
        im_name=FLAGS.im_name, n_channel=3,
        data_dir=IM_PATH, batch_size=1, rescale=False)

    # Create a testing GoogLeNet model
    test_model = GoogLeNet_cifar(
        n_channel=3, n_class=10, bn=True, sub_imagenet_mean=False)
    test_model.create_test_model()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, '{}inception-cifar-epoch-{}'.format(FLAGS.savePath, FLAGS.load))
        while image_data.epochs_completed < 1:
            # read batch files
            batch_data = image_data.next_batch_dict()
            # get batch file names
            batch_file_name = image_data.get_batch_file_name()[0]
            # get prediction results
            pred = sess.run(test_model.layers['top_5'],
                            feed_dict={test_model.image: batch_data['image']})
            # display results
            for re_prob, re_label, file_name in zip(pred[0], pred[1], batch_file_name):
                print('===============================')
                print('[image]: {}'.format(file_name))
                for i in range(5):
                    print('{}: probability: {:.02f}, label: {}'
                          .format(i+1, re_prob[i], label_dict[re_label[i]]))

if __name__ == "__main__":
    FLAGS = get_args()

    if FLAGS.train:
        train()
    if FLAGS.eval:
        evaluate()
    if FLAGS.predict:
        predict()
