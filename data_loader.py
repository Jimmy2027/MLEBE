import os
import nibabel as nib
from matplotlib import pyplot as plt
import numpy as np
from skimage import transform as tf

from utils import *

not_preprocessed_dir = '/usr/share/'
image_dir = '/Users/Hendrik/Documents/mlebe_data/preprocessed'
image_dir_remote = '/mnt/scratch/'


def load_bidsdata():
    for o in os.listdir(not_preprocessed_dir):
        if not o.startswith('irsabi'):
                for root, dirs, files in os.walk(os.path.join(image_dir_remote, o)):
                    for file in files:
                        if file.endswith("_T2w.nii.gz"):
                            print(root)



def load_img_remote():
    visualisation = False

    im_data = []
    for o in os.listdir(image_dir_remote):
        if o != 'irsabi':
            for x in os.listdir(os.path.join(image_dir_remote, o)):
                if x.endswith('preprocessing'):
                    for root, dirs, files in os.walk(os.path.join(image_dir_remote, o, x)):
                        for file in files:
                            if file.endswith("_T2w.nii.gz"):
                                im_data.append(os.path.join(root, file))



    im_data = np.sort(im_data)
    data = []
    for i in im_data:
        img = nib.load(i)
        img_data = img.get_data()
        temp = np.moveaxis(img_data,2,0)
        img_data = pad_img(temp)
        img_data = data_normalization(img_data)
        path = os.path.join('visualisation', os.path.basename(i), 'padded_data')
        if visualisation == True:
            save_img(img_data, path)
            visualisation = False

        data.append(img_data)
    return data

def load_img(visualisation):

    im_data = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("TurboRARE_T2w.nii.gz"):
                im_data.append(os.path.join(root, file))

    im_data = np.sort(im_data)
    data = []
    for i in im_data:
        img = nib.load(i)
        img_data = img.get_data()
        temp = np.moveaxis(img_data,2,0)
        img_data = pad_img(temp)
        img_data = data_normalization(img_data)
        path = os.path.join('visualisation', os.path.basename(i), 'padded_data')
        if visualisation == True:
            save_img(img_data, path)
            visualisation = False

        data.append(img_data)
    return data


def load_mask(data_dir, visualisation):

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
        temp = np.moveaxis(img_data,2,0)
        img_data = pad_img(temp)
        img_data = data_normalization(img_data)
        path = os.path.join('visualisation', os.path.basename(i), 'padded_data')
        if visualisation == True:
            save_img(img_data, path)
            visualisation = False

        data.append(img_data)


    return data


def pad_img(img):
    shape = (64, 128)
    padded = np.empty((img.shape[0], shape[0], shape[1]))
    for i in range(img.shape[0]):
        padd_y = shape[0] - img.shape[1]
        padd_x = shape[1] - img.shape[2]
        padded[i, ...] = np.pad(img[i, ...], ((padd_y//2, shape[0]-padd_y//2-img.shape[1]), (padd_x//2, shape[1]-padd_x//2-img.shape[2])), 'constant')
    return padded


def resize(img):
    shape = (256, 256)
    padded = np.empty((shape[0], shape[1], img.shape[2]))
    for i in range (img.shape[2]):
        padded[...,i] = tf.resize(img[..., i]/np.max(img[..., i]), output_shape = shape, mode = 'constant')       #Todo am normalizing the data here
    return padded


if __name__ == '__main__':      #only gets called if Unet.py is run
    load_bidsdata()