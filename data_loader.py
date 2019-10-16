import os
import nibabel as nib
from matplotlib import pyplot as plt
import numpy as np

data_dir = '/Users/Hendrik/Documents/mlebe_data/mouse-brain-atlases/'


def load_img():
    img = []
    im_data = []
    for o in os.listdir(data_dir):
        if o.endswith('200micron.nii'):
            im_data.append(os.path.join(data_dir,o))

    im_data = np.sort(im_data)
    data = []
    print(im_data)
    for i in im_data:
        img = nib.load(i)
        img_data = img.get_data()
        img_data = pad_img(img_data)
        img_data = np.expand_dims(img_data,-1)
        data.append(img_data)

    for i in range(data[0][1, 1].shape[0]):
        plt.imshow(data[0][..., i,0], cmap='gray')
        plt.savefig('visualisation/img_{}.pdf'.format(i))

    return data


def load_mask():
    mask = []
    im_data = []
    for o in os.listdir(data_dir):
        if o.endswith('200micron_mask.nii'):
            im_data.append(os.path.join(data_dir,o))

    data = []
    im_data = np.sort(im_data)
    print(im_data)
    for i in im_data:
        img = nib.load(i)
        img_data = img.get_data()
        img_data = np.fliplr(img_data)          #todo masks are flipped?
        img_data = pad_img(img_data)
        img_data = np.expand_dims(img_data,-1)
        data.append(img_data)

    for i in range(data[0][1, 1].shape[0]):
        plt.imshow(data[0][..., i,0], cmap='gray')
        plt.savefig('visualisation/mask_{}.pdf'.format(i))

    return data


def pad_img(img):
    shape = (256, 256)
    padded = np.empty((shape[0],shape[1], img.shape[2]))
    for i in range(img.shape[2]):
        padd_y = shape[0] - img.shape[0]
        padd_x = shape[1] - img.shape[1]
        padded[...,i] = np.pad(img[..., i], ((padd_y//2, shape[0]-padd_y//2-img.shape[0]), (padd_x//2, shape[1]-padd_x//2-img.shape[1])),'constant')

        # plt.imshow(padded[...,i], cmap='gray')
        # plt.savefig('visualisation/padded/padded_{}.pdf'.format(i))
    return padded
