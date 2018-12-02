import os
import pprint

import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from keras.utils import Sequence


def compare_old_new_data(old_data_dir,new_data_dir):
    old_csv = os.path.join(old_data_dir,'train.csv')
    new_csv = os.path.join(new_data_dir,'train.csv')

    old_pd = pd.read_csv(old_csv)
    new_pd = pd.read_csv(new_csv)

    old_images = set(old_pd.Image.values)
    new_images = set(new_pd.Image.values)

    print('Old set has {} images'.format(len(old_images)))
    print('Old set has, but new set does not: ')
    pprint.pprint(len(old_images-new_images))

    print('New set has {} images'.format(len(new_images)))
    print('New set has, but old set does not: ')
    pprint.pprint(len(new_images-old_images))

    old_classes = set(old_pd.Id.values)
    new_classes = set(old_pd.Id.values)

    print('Old set has {} classes'.format(len(old_classes)))
    print('Old set has, but new set does not: ')
    pprint.pprint(len(old_classes - new_classes))

    print('New set has {} classes'.format(len(new_classes)))
    print('New set has, but old set does not: ')
    pprint.pprint(len(new_classes - old_classes))

    new_pd['Image'] = new_pd['Image'].apply(lambda X:new_data_dir+'/'+X)
    old_pd['Image'] = old_pd['Image'].apply(lambda X:old_data_dir+'/'+X)

    total_pd = new_pd.append(old_pd,ignore_index=True)

    return total_pd



def get_class_name(data_frame):
    class_dict = {}

    for i in data_frame.Id.values:
        if i in class_dict.keys():
            class_dict[i] +=1
        else:
            class_dict[i] = 1
    return class_dict

def sparse_datacsv(data_dir,train_ratio,random_seed=666):
    """

    :param data_dir:
    :param train_ratio:
    :param random_seed:
    :return:
    """
    trainval_csv = os.path.join(data_dir, 'train.csv')
    test_csv = os.path.join(data_dir, 'sample_submission.csv')

    trainval_pd = pd.read_csv(trainval_csv)
    trainval_class_dict = get_class_name(trainval_pd)

    print('Total data has {} classes: '.format(len(trainval_class_dict.keys())))
    pprint.pprint(trainval_class_dict)

    test_pd = pd.read_csv(test_csv)
    train, val = train_test_split(trainval_pd, train_size=train_ratio, random_state=random_seed, shuffle=True)

    train_class_dict = get_class_name(train)
    print('Total train data has {} classes: '.format(len(train_class_dict.keys())))
    pprint.pprint(train_class_dict)

    train_dict = {k: v for k, v in zip(train.Image.values, train.Id.values)}
    val_dict = {k: v for k, v in zip(val.Image.values, val.Id.values)}
    test_dict = {k: None for k in test_pd.Image.values}

    className = trainval_class_dict.keys()

    return train_dict,val_dict,test_dict,className





class DataGen(Sequence):
    def __init__(self,data_dict):
        """

        :param data_path:
        """

    def __getitem__(self, index):
        pass


    def __len__(self):
        pass


    def on_epoch_end(self):

        pass

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item
