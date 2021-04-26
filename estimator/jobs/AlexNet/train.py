# -*- coding: utf-8 -*-

import tensorflow as tf

from alexnet import AlexNet
from dataset_helper import read_cifar_10
import sys

INPUT_WIDTH = 70
INPUT_HEIGHT = 70
INPUT_CHANNELS = 3

NUM_CLASSES = 10

LEARNING_RATE = 0.001   # Original value: 0.01
MOMENTUM = 0.9
KEEP_PROB = 0.5

EPOCHS = int(sys.argv[1])
BATCH_SIZE = 128

use_cpu = sys.argv[2] == "--cpu"
hardware_param = int(sys.argv[4])

import importlib.util
spec = importlib.util.spec_from_file_location("time_writter", "../../time_writter.py")
time_writter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(time_writter)

time_writter.Start()

print('Reading CIFAR-10...',flush=True)
X_train, Y_train, X_test, Y_test = read_cifar_10(image_width=INPUT_WIDTH, image_height=INPUT_HEIGHT)

alexnet = AlexNet(input_width=INPUT_WIDTH, input_height=INPUT_HEIGHT, input_channels=INPUT_CHANNELS,
                  num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE, momentum=MOMENTUM, keep_prob=KEEP_PROB)

# Parse hardware
config = None
if use_cpu:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    config=tf.ConfigProto(inter_op_parallelism_threads=hardware_param,
               intra_op_parallelism_threads=hardware_param,
               device_count={'GPU':0, 'CPU':1})
    print("Using CPU",flush=True)
else:
    config=tf.ConfigProto(device_count={'GPU':hardware_param, 'CPU':1})
    visible_gpus = ''
    for gpu in range(hardware_param):
        visible_gpus += str(gpu) + ","
    config.gpu_options.visible_device_list=visible_gpus[:-1] # remove last comma
    print("Using GPU",flush=True)

with tf.Session(config=config) as sess:

    file_writer = tf.summary.FileWriter(logdir='./log', graph=sess.graph)

    summary_operation = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())

    for i in range(EPOCHS):
        time_writter.LogUpdate()
        print("Epoch " + str(i) + " started.",flush=True)
        print(time_writter.GetResults(),flush=True)

        alexnet.train_epoch(sess, X_train, Y_train, BATCH_SIZE, file_writer, summary_operation, i)

    time_writter.LogUpdate()
    final_train_accuracy = alexnet.evaluate(sess, X_train, Y_train, BATCH_SIZE)
    final_test_accuracy = alexnet.evaluate(sess, X_test, Y_test, BATCH_SIZE)

    print('Final Train Accuracy = {:.3f}'.format(final_train_accuracy))
    print('Final Test Accuracy = {:.3f}'.format(final_test_accuracy))
    print()

    alexnet.save(sess, './model/alexnet')
    print('Model saved.')
    print()

print('Training done successfully.')

time_writter.LogUpdate()
print(time_writter.GetResults())