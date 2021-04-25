
#import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import tutorials.mnist.input_data as input_data
import config as cfg
import os
import lenet
from lenet import Lenet
import sys

import importlib.util
spec = importlib.util.spec_from_file_location("time_writter", "../../time_writter.py")
time_writter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(time_writter)

use_cpu = sys.argv[1] == "--cpu"
hardware_param = int(sys.argv[3])

def main():
    time_writter.Start()

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Configure hardware
    config = None
    if use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"
        config=tf.ConfigProto(inter_op_parallelism_threads=hardware_param,
                   intra_op_parallelism_threads=hardware_param,
                   device_count={'GPU':0, 'CPU':1})
        print("Using CPU")
    else:
        config=tf.ConfigProto(device_count={'GPU':hardware_param, 'CPU':1})
        visible_gpus = ''
        for gpu in range(hardware_param):
            visible_gpus += str(gpu) + ","
        config.gpu_options.visible_device_list=visible_gpus[:-1] # remove last comma
        print("Using GPU")

    sess = tf.Session(config=config)
    batch_size = cfg.BATCH_SIZE
    parameter_path = cfg.PARAMETER_FILE
    lenet = Lenet()
    max_iter = cfg.MAX_ITER


    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())

    with open("../../results/" + os.environ['JOB_NAME'], "w") as f:
        f.write("Starting experiment.")
        print("Starting experiments.")
        f.flush()
        for i in range(max_iter):
            time_writter.LogUpdate()
            if i % 500 == 0:
                print("Epoch " + str(i) + " completed.")
                print(time_writter.GetResults())
                f.write(time_writter.GetResults() + "\n")
                f.flush()

            batch = mnist.train.next_batch(batch_size)
            sess.run(lenet.train_op,feed_dict={lenet.raw_input_image: batch[0],lenet.raw_input_label: batch[1]})

        time_writter.LogUpdate()
        save_path = saver.save(sess, parameter_path)

        time_writter.LogUpdate()
        print(time_writter.GetResults())
        f.write(time_writter.GetResults())

if __name__ == '__main__':
    main()


