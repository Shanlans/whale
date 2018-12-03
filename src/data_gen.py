import os
import pickle
import numpy as np
import pandas as pd

from math import *
from collections import defaultdict
from imagehash import phash
from PIL import Image as pil_image
from tqdm import tqdm


from keras.utils import Sequence,to_categorical



def phash_file(data_dir,data_frame,train_set=True):

    def p2h(data_frame):
        data = data_frame.Image.values

        img_hash = {}

        for p in tqdm(data, desc='Doing Phash...', total=len(data)):
            img = pil_image.open(p)
            h = phash(img)
            img_hash[p] = h
        return img_hash

    if os.path.exists(os.path.join(data_dir,'train_phash.pickle')) and train_set:
        with open(os.path.join(data_dir,'train_phash.pickle'),'rb') as f:
            img_hash = pickle.load(f)
    elif os.path.exists(os.path.join(data_dir,'test_phash.pickle')) and not train_set:
        with open(os.path.join(data_dir,'test_phash.pickle'),'rb') as f:
            img_hash = pickle.load(f)
    else:
        '''
        Cal each image's hash
        '''
        img_hash = p2h(data_frame)

        if train_set:
            with open(os.path.join(data_dir,'train_phash.pickle'),'wb') as f:
                pickle.dump(img_hash,f,protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(os.path.join(data_dir,'test_phash.pickle'),'wb') as f:
                pickle.dump(img_hash,f,protocol=pickle.HIGHEST_PROTOCOL)

    def h2p(img_hash):
        # Convert {Image:hash} -> {hash:image}, check whether has same hash for different images
        h2ps = {}
        for p, h in img_hash.items():
            if h not in h2ps:
                h2ps[h] = []
            if p not in h2ps[h]:
                h2ps[h].append(p)
        return h2ps


    '''
    Check how many same hash dataset has 
    '''
    h2ps = h2p(img_hash)

    def match(h1, h2):
        # Two phash values are considered duplicate if, for all associated image pairs:
        # 1) They have the same mode and size;
        # 2) After normalizing the pixel to zero mean and variance 1.0, the mean square error does not exceed 0.1
        for p1 in h2ps[h1]:
            for p2 in h2ps[h2]:
                i1 = pil_image.open(p1)
                i2 = pil_image.open(p2)
                if i1.mode != i2.mode or i1.size != i2.size: return False
                a1 = np.array(i1)
                a1 = a1 - a1.mean()
                a1 = a1 / sqrt((a1 ** 2).mean())
                a2 = np.array(i2)
                a2 = a2 - a2.mean()
                a2 = a2 / sqrt((a2 ** 2).mean())
                a = ((a1 - a2) ** 2).mean()
                if a > 0.1: return False
        return True

    hs = list(h2ps.keys())

    # If the images are close enough, associate the two phash values (this is the slow part: n^2 algorithm)
    h2h = {}
    if os.path.exists(os.path.join(data_dir,'train_phash_sim.pickle')) and train_set:
        with open(os.path.join(data_dir,'train_phash_sim.pickle'),'rb') as f:
            h2h = pickle.load(f)
    elif os.path.exists(os.path.join(data_dir,'test_phash_sim.pickle')) and not train_set:
        with open(os.path.join(data_dir,'test_phash_sim.pickle'),'rb') as f:
            h2h = pickle.load(f)
    else:
        '''
        check image hashes similarity between each other, mapping the similar hash to their larger one 
        '''
        for i, h1 in enumerate(tqdm(hs, desc='Analyzing each hash similarity', total=len(hs))):
            for h2 in hs[:i]:
                if h1 - h2 <= 6 and match(h1, h2):
                    s1 = str(h1)
                    s2 = str(h2)
                    if s1 < s2:
                        s1, s2 = s2, s1
                    h2h[s1] = s2
        if train_set:
            with open(os.path.join(data_dir,'train_phash_sim.pickle'),'wb') as f:
                pickle.dump(h2h,f,protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(os.path.join(data_dir,'test_phash_sim.pickle'),'wb') as f:
                pickle.dump(h2h,f,protocol=pickle.HIGHEST_PROTOCOL)

    '''
    Now, update the images new hash, the next step will eliminate the more hash  
    '''
    for p, h in img_hash.items():
        h = str(h)
        if h in h2h: h = h2h[h]
        img_hash[p] = h

    '''
    Update how many same hash dataset has
    '''
    h2ps = {}
    for p, h in img_hash.items():
        if h not in h2ps:
            h2ps[h] = []
        if p not in h2ps[h]:
            h2ps[h].append(p)

    for p,h in h2ps.items():
        if len(h)>1:
            print(h)


    return img_hash

def compare_old_new_data(new_csv):
    """
    Merge new data set with old data set
    !To do: Simply merge, I think it need to do improvement
    :param new_csv:
    :return:
    """
    old_data_dir = './last_data'
    old_csv = os.path.join(old_data_dir,'train.csv')

    old_pd = pd.read_csv(old_csv)
    new_pd = pd.read_csv(new_csv)

    old_images = set(old_pd.Image.values)
    new_images = set(new_pd.Image.values)

    print('Old set has {} images'.format(len(old_images)))
    print('Old set has, but new set does not: ')
    print(len(old_images-new_images))

    print('New set has {} images'.format(len(new_images)))
    print('New set has, but old set does not: ')
    print(len(new_images-old_images))

    old_classes = set(old_pd.Id.values)
    new_classes = set(old_pd.Id.values)

    print('Old set has {} classes'.format(len(old_classes)))
    print('Old set has, but new set does not: ')
    print(len(old_classes - new_classes))

    print('New set has {} classes'.format(len(new_classes)))
    print('New set has, but old set does not: ')
    print(len(new_classes - old_classes))

    new_pd['Image'] = new_pd['Image'].apply(lambda X:'./data'+'/train/'+X)
    old_pd['Image'] = old_pd['Image'].apply(lambda X:old_data_dir+'/train/'+X)

    total_pd = new_pd.append(old_pd,ignore_index=True)

    return total_pd



class Sparse_data(object):

    def __init__(self,data_dir,batch_size,random_seed,other_class='new_whale'):
        """

        :param data_dir:
        :param random_seed:
        :param batch_size:
        :param other_class:
        """

        self.other_class = other_class
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.data_csv = os.path.join(data_dir, 'train.csv')

        data_frame = pd.read_csv(self.data_csv)
        self.data_frame['Image'] = data_frame['Image'].apply(lambda X: data_dir + '/train/' + X)
        self.image_to_class = dict(zip(data_frame.Image.values,data_frame.Id.values))


        self.class_to_id = {}
        self.image_to_id = {}
        self.class_num = {}
        self.class_has_images = defaultdict(list)
        self.other_class_has_images = {self.other_class:[]}

        self._sparse_data()

    def _sparse_data(self):

        self.class_to_id = {k:i for i,k in enumerate(self.image_to_class.values())}

        self.image_to_id = {k:self.class_to_id[v] for k,v in self.image_to_class.items()}

        for img,cls in self.image_to_class.items():

            # if cls == self.other_class:
            #     self.other_class_has_images[self.other_class].append(img)
            # else:
            self.class_has_images[cls].append(img)

        self.class_to_num = {k:len(v)for k,v in self.class_has_images.items()}

        self.class_weight = np.array([len(self.class_has_images[class_]) for class_ in self.list_classes])

    def _get_sample(self):
        class_idx = np.random.choice(self.range_list_classes, 1, p=self.class_weight)[0]

    def gen(self,batch_size):
        while True:
            list_positive_examples_1 = []
            list_negative_examples = []
            list_positive_examples_2 = []

            for i in range(batch_size):
                positive_example_1, negative_example, positive_example_2 = triplet_gen.get_sample()
                path_pos1 = join(path_train, positive_example_1)
                path_neg = join(path_train, negative_example)
                path_pos2 = join(path_train, positive_example_2)

                positive_example_1_img = read_and_resize(path_pos1)
                negative_example_img = read_and_resize(path_neg)
                positive_example_2_img = read_and_resize(path_pos2)

                positive_example_1_img = augment(positive_example_1_img)
                negative_example_img = augment(negative_example_img)
                positive_example_2_img = augment(positive_example_2_img)

                list_positive_examples_1.append(positive_example_1_img)
                list_negative_examples.append(negative_example_img)
                list_positive_examples_2.append(positive_example_2_img)

            A = preprocess_input(np.array(list_positive_examples_1))
            B = preprocess_input(np.array(list_positive_examples_2))
            C = preprocess_input(np.array(list_negative_examples))

            label = None

            yield ({'anchor_input': A, 'positive_input': B, 'negative_input': C}, label)