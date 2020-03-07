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


def load_img_remote(image_dir_remote, blacklist, test = False):
    print('*** Loading images ***')
    im_data = []
    for o in os.listdir(image_dir_remote):
        if o != 'irsabi' and o != 'results' and not o.startswith('mlebe'):
            print(o)
            for x in os.listdir(os.path.join(image_dir_remote, o)):
                if x.endswith('preprocessing') or x.startswith('preprocess') and not x.endswith('work'):
                    for root, dirs, files in os.walk(os.path.join(image_dir_remote, o, x)):
                        for file in files:
                            if file.endswith("_T2w.nii.gz"):
                                if not blacklist == False:
                                    blacklisted = False
                                    for i in blacklist:
                                        if file.startswith('sub-' + i.subj + '_ses-' + i.sess + '_'):
                                            blacklisted = True
                                            print('blacklisted found: {}'.format(file))

                                    if blacklisted == False:
                                        im_data.append(os.path.join(root, file))
                                else:
                                    im_data.append(os.path.join(root, file))


    im_data = np.sort(im_data)
    print('*** Loading {} subjects ***'.format(len(im_data)))

    im_data = list(dict.fromkeys(im_data))
    data = []
    for i in im_data:
        img = nib.load(i)
        data.append(img)

    if test:
        data = data[:10]

    return data


def load_mask(data_dir):
    print('*** Loading mask ***')
    im_data = []
    for o in os.listdir(data_dir):
        if o == 'dsurqec_200micron_mask.nii':
            im_data.append(os.path.join(data_dir,o))

    data = []
    im_data = np.sort(im_data)

    for i in im_data:
        img = nib.load(i)
        data.append(img)
    return data

def load_func_img(image_dir_remote, test = False):
    from nipype.interfaces import fsl
    print('*** Loading images ***')
    func_training_dir = os.path.abspath(os.path.expanduser('/var/tmp/func_training'))

    if not os.path.exists(func_training_dir):
        print('creating dir: ', func_training_dir)
        os.makedirs(func_training_dir)
    im_data = []
    for o in os.listdir(image_dir_remote):
        if o != 'irsabi' and not o.startswith('.'):
            for x in os.listdir(os.path.join(image_dir_remote, o)):
                if x.endswith('preprocessing'):
                    for root, dirs, files in os.walk(os.path.join(image_dir_remote, o, x)):
                            if root.endswith('func'):
                                for file in files:
                                    if file.endswith(".nii.gz"):
                                        tMean_path = os.path.join(func_training_dir, 'tMean_' + file)
                                        if not os.path.isfile(tMean_path):
                                            command = 'fslmaths {a} -Tmean {b}'.format(a = os.path.join(root, file), b = tMean_path)
                                            print(command)
                                            os.system(command)
                                        im_data.append(tMean_path)

    im_data = np.sort(im_data)
    print('*** Loading {} subjects ***'.format(len(im_data)))

    if test == True:
        im_data = im_data[:1]

    data = []
    for i in im_data:
        img = nib.load(i)
        data.append(img)

    return data


if __name__ == '__main__':
    load_bidsdata()