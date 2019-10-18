import os
import nibabel as nib
from matplotlib import pyplot as plt
import numpy as np
from skimage import transform as tf

from utils import *

image_dir = '/Users/Hendrik/Documents/mlebe_data/preprocessed'
data_dir = '/Users/Hendrik/Documents/mlebe_data/mouse-brain-atlases/'   #local
# data_dir = '/usr/share/mouse-brain-atlases/'    #remote

visualisation = False

def load_img():
    visualisation = False

    im_data = []
    for root, dirs, files in os.walk("/Users/Hendrik/Documents/mlebe_data/preprocessed"):
        for file in files:
            if file.endswith("TurboRARE_T2w.nii.gz"):
                im_data.append(os.path.join(root, file))

    for o in os.listdir(image_dir):
        if o.endswith('200micron.nii'):
            im_data.append(os.path.join(data_dir, o))

    im_data = np.sort(im_data)
    data = []
    for i in im_data:
        img = nib.load(i)
        img_data = img.get_data()
        path = os.path.join('visualisation', os.path.basename(i), 'untouched_data')
        if visualisation == True:
            save_img(img_data, path)
            visualisation = False

        img_data = np.expand_dims(img_data,-1)
        data.append(img_data)
    return data


def load_mask():
    visualisation = False

    mask = []
    im_data = []
    for o in os.listdir(data_dir):
        if o == 'dsurqec_200micron_mask.nii':
            im_data.append(os.path.join(data_dir,o))

    data = []
    im_data = np.sort(im_data)
    print(im_data)
    for i in im_data:
        img = nib.load(i)
        img_data = img.get_data()
        path = os.path.join('visualisation', os.path.basename(i), 'untouched_data')
        if visualisation == True:
            save_img(img_data, path)
            visualisation = False
        img_data = np.fliplr(img_data)          #todo masks are flipped?

        img_data = np.expand_dims(img_data, -1)
        data.append(img_data)


    return data


def pad_img(img):
    shape = (256, 256)
    padded = np.empty((shape[0],shape[1], img.shape[2]))
    for i in range(img.shape[2]):
        padd_y = shape[0] - img.shape[0]
        padd_x = shape[1] - img.shape[1]
        padded[..., i] = np.pad(img[..., i], ((padd_y//2, shape[0]-padd_y//2-img.shape[0]), (padd_x//2, shape[1]-padd_x//2-img.shape[1])), 'constant')
    return padded


def resize(img):
    shape = (256, 256)
    padded = np.empty((shape[0],shape[1], img.shape[2]))
    for i in range (img.shape[2]):
        padded[...,i] = tf.resize(img[..., i]/np.max(img[..., i]), output_shape = shape, mode = 'constant')       #Todo am normalizing the data here
    return padded
