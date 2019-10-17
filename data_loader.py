import os
import nibabel as nib
from matplotlib import pyplot as plt
import numpy as np
from skimage import transform as tf

# data_dir = '/Users/Hendrik/Documents/mlebe_data/mouse-brain-atlases/'   #local
data_dir = '/usr/share/mouse-brain-atlases/'    #remote

visualisation = False

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

        if visualisation ==True:
            if not os.path.exists(os.path.join('visualisation', os.path.basename(i), 'untouched_data')):
                os.makedirs(os.path.join('visualisation', os.path.basename(i),  'untouched_data'))
            for j in range(img_data.shape[2]):
                plt.imshow(img_data[..., j], cmap='gray')
                plt.savefig('visualisation/'+os.path.basename(i)+'/untouched_data/img_{}.pdf'.format(j))

        img_data = resize(img_data)     #resizes the data to (256*256) AND normalizes it        #todo need to reshape data?

        img_data = np.expand_dims(img_data,-1)
        data.append(img_data)

        if visualisation ==True:
            if not os.path.exists(os.path.join('visualisation', os.path.basename(i), 'resized')):
                os.makedirs(os.path.join('visualisation', os.path.basename(i),  'resized'))
            for j in range(img_data.shape[2]):
                    plt.imshow(img_data[..., j, 0], cmap='gray')
                    plt.savefig('visualisation/' + os.path.basename(i) + '/resized/img_{}.pdf'.format(j))

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

        if visualisation ==True:
            if not os.path.exists(os.path.join('visualisation', os.path.basename(i), 'untouched_data')):
                os.makedirs(os.path.join('visualisation', os.path.basename(i),  'untouched_data'))
            for j in range(img_data.shape[2]):
                plt.imshow(img_data[..., j], cmap='gray')
                plt.savefig('visualisation/'+os.path.basename(i)+'/untouched_data/img_{}.pdf'.format(j))

        img_data = np.fliplr(img_data)          #todo masks are flipped?

        img_data = resize(img_data)         #resizes the data to (256*256) AND normalizes it
        img_data = np.expand_dims(img_data, -1)
        data.append(img_data)

        if visualisation ==True:
            if not os.path.exists(os.path.join('visualisation', os.path.basename(i), 'resized')):
                os.makedirs(os.path.join('visualisation', os.path.basename(i), 'resized'))
            for j in range(img_data.shape[2]):
                    plt.imshow(img_data[..., j, 0], cmap='gray')
                    plt.savefig('visualisation/' + os.path.basename(i) + '/resized/img_{}.pdf'.format(j))

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
        padded[...,i] = tf.resize(img[...,i]/np.max(img[...,i]), output_shape = shape, mode = 'constant')       #Todo am normalizing the data here
    return padded
