import os

import tensorflow as tf

import numpy as np


from src import *




flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir','./data','dataset folder')
flags.DEFINE_integer('random_seed',666,'random seed for np,sklearn or others')

# Training setting
flags.DEFINE_float('train_ratio',0.7,'Train set ratio')




def main(argv):

    data_dir = FLAGS.data_dir
    train_ratio = FLAGS.train_ratio
    random_seed = FLAGS.random_seed

    train_dict,val_dict,test_dict,class_name = sparse_datacsv(data_dir=data_dir,
                                                     train_ratio=train_ratio,
                                                    expand_training_data=False,
                                                    random_seed=random_seed)


if __name__ == "__main__":
    tf.app.run()






