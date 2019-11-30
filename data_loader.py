import os
import nibabel as nib
from matplotlib import pyplot as plt
import numpy as np

from utils import *

not_preprocessed_dir = '/usr/share/'


def load_bidsdata():
    """

    :return: list of paths of all the bids files
    """
    paths = []
    filess = []
    print('*** Loading bidsdata ***')
    for o in os.listdir(not_preprocessed_dir):
        if not o.startswith('irsabi') and o.endswith('bidsdata'):
            for root, dirs, files in os.walk(os.path.join(not_preprocessed_dir, o)):
                for file in files:
                    if file.endswith("_T2w.nii.gz"):
                        print(os.path.join(not_preprocessed_dir, o, root, file))
                        paths.append(os.path.join(not_preprocessed_dir, o, root, file))
                        filess.append(file)
    return paths, filess


def load_img_remote(image_dir_remote, blacklist):
    print('*** Loading images ***')
    im_data = []
    for o in os.listdir(image_dir_remote):
        if o != 'irsabi':
            for x in os.listdir(os.path.join(image_dir_remote, o)):
                if x.endswith('preprocessing'):
                    for root, dirs, files in os.walk(os.path.join(image_dir_remote, o, x)):
                        for file in files:
                            if file.endswith("_T2w.nii.gz"):
                                blacklisted = False
                                for i in blacklist:
                                    partial_blacklisted = True
                                    while partial_blacklisted == True:
                                        for j in i:
                                            if (j in file) == True:
                                                partial_blacklisted = True
                                            else:
                                                partial_blacklisted = False

                                    if partial_blacklisted == True:
                                        blacklisted == True
                                        print('blacklisted found: {}'.format(i))
                                if blacklisted == False:
                                   im_data.append(os.path.join(root, file))



    im_data = np.sort(im_data)
    print('*** Loading {} subjects ***'.format(len(im_data)))

    data = []
    for i in im_data:
        img = nib.load(i)
        # img_data = img.get_data()
        # temp = np.moveaxis(img_data,2,0)
        # img_data = pad_img(temp, shape)
        # img_data = data_normalization(img_data)


        data.append(img)
    return data

def load_img(image_dir, blacklist):
    print('*** Loading images ***')

    im_data = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("TurboRARE_T2w.nii.gz"):
                blacklisted = False
                for i in blacklist:
                    if all(elem in file for elem in i) == True:
                        blacklisted = True
                        print('blacklisted found: {}'.format(file))
                    # partial_blacklisted = True
                    # while partial_blacklisted == True:
                    #     for j in i:
                    #         if (j in file) == True:
                    #             partial_blacklisted = True
                    #         else:
                    #             partial_blacklisted = False

                    # if partial_blacklisted == True:
                    #     blacklisted == True
                    #     print('blacklisted found: {}'.format(i))
                if blacklisted == False:
                    im_data.append(os.path.join(root, file))

    im_data = np.sort(im_data)
    print('*** Loading {} subjects ***'.format(len(im_data)))

    data = []

    for i in im_data:
        img = nib.load(i)
        # img_data = img.get_data()   #shape = (63, 96, 48)
        # temp = np.moveaxis(img_data,2,0)    #shape = (48, 63, 96)
        # img_data = pad_img(temp, shape)
        # img_data = data_normalization(img_data)
        # path = os.path.join('visualisation', os.path.basename(i), 'padded_data')
        # if visualisation == True:
        #     save_img(img_data, path)
        #     visualisation = False

        data.append(img)
    return data


def load_mask(data_dir):
    print('*** Loading mask ***')
    mask = []
    im_data = []
    for o in os.listdir(data_dir):
        if o == 'dsurqec_200micron_mask.nii':
            im_data.append(os.path.join(data_dir,o))

    data = []
    im_data = np.sort(im_data)

    for i in im_data:
        img = nib.load(i)
        # img_data = img.get_data()
        # temp = np.moveaxis(img_data,2,0)
        # img_data = pad_img(temp, shape)
        # img_data = data_normalization(img_data)
        # path = os.path.join('visualisation', os.path.basename(i), 'padded_data')
        # if visualisation == True:
        #     save_img(img_data, path)
        #     visualisation = False

        data.append(img)


    return data





if __name__ == '__main__':
    load_bidsdata()