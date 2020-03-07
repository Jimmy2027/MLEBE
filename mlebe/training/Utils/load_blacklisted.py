import data_loader as dl
import os
import utils
import numpy as np
import nibabel as nib
import copy

def load_blacklisted(remote, slice_view, shape, save_dir):
    """

    :return: returns array of blacklisted slices
    """
    blacklist = utils.write_slice_blacklist()
    if remote == 'hongg' or remote == True:
        image_dir_remote = '/mnt/scratch/'
        data_dir = '/usr/share/mouse-brain-atlases/'


    elif remote == 'leonhard':
        image_dir_remote = '/cluster/scratch/klugh/mlebe_scratch/'
        data_dir = '/cluster/scratch/klugh/mouse-brain-atlases/'



    elif remote == 'epfl':
        image_dir_remote = '/home/klug/Hendrik/MLEBE/mlebe_scratch/'
        data_dir = '/home/klug/Hendrik/MLEBE/mouse-brain-atlases/'



    else:
        image_dir = '/Users/Hendrik/Documents/mlebe_data/preprocessed'
        if os.path.exists('/Volumes/something/mlebe_scratch'):

            image_dir = '/Volumes/something/mlebe_scratch'

        data_dir = '/Users/Hendrik/Documents/mlebe_data/mouse-brain-atlases/'  # local


    temp = dl.load_mask(data_dir)
    mask = temp[0].get_data()



    im_data = []
    slices = []     #stores the slice numbers that are blacklisted
    if remote == True:
        for o in os.listdir(image_dir_remote):
            if o != 'irsabi':
                for x in os.listdir(os.path.join(image_dir_remote, o)):
                    if x.endswith('preprocessing'):
                        for root, dirs, files in os.walk(os.path.join(image_dir_remote, o, x)):
                            for file in files:
                                for bfile in blacklist:
                                    if file == bfile.filename:
                                        img_temp = nib.load(os.path.join(root, file))
                                        slices.append(int(bfile.slice))
                                        im_data.append(img_temp.get_data())

    if remote == False:
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                for bfile in blacklist:
                    if file == bfile.filename:
                        img_temp = nib.load(os.path.join(root, file))
                        slices.append(int(bfile.slice))
                        im_data.append(img_temp.get_data())

    image_list = []
    mask_list = []
    image_list_temp = []
    mask_list_temp = []
    for it, img in enumerate(im_data):
        if slice_view == 'coronal':
            img_temp = np.moveaxis(img, 1, 0)
            mask_temp = np.moveaxis(copy.deepcopy(mask), 1, 0)
        elif slice_view == 'axial':
            img_temp = np.moveaxis(img, 2, 0)
            mask_temp = np.moveaxis(copy.deepcopy(mask), 2, 0)
        fitted_mask = utils.arrange_mask(img_temp, mask_temp)
        img_temp, fitted_mask = utils.remove_black_images(img_temp, fitted_mask)
        img_preprocessed = utils.preprocess(img_temp, shape, slice_view, switched_axis= True)
        mask_preprocessed = utils.preprocess(fitted_mask, shape, slice_view, switched_axis = True)
        image_list_temp.append(img_preprocessed)
        mask_list_temp.append(mask_preprocessed)



        try:
            image_list.append(img_preprocessed[slices[it]])
            mask_list.append(mask_preprocessed[slices[it]])
        except Exception as e:
            print(e)

    # # utils.save_datavisualisation([image_list_temp,mask_list_temp], save_dir + 'test/')
    # utils.save_datavisualisation_plt([image_list_temp, mask_list_temp], save_dir + 'test/')
    return image_list, mask_list