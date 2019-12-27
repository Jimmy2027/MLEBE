import data_loader as dl
import os
import utils
import copy
import nibabel as nib

def load_blacklisted(remote, slice_view, shape):
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

    mask = utils.preprocess(dl.load_mask(data_dir), shape, slice_view)


    im_data = []
    slices=[]
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
                                        slices.append(bfile.slice)
                                        im_data.append(img_temp)

    if remote == False:
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                for bfile in blacklist:
                    if file == bfile.filename:
                        img_temp = nib.load(os.path.join(root, file))
                        slices.append(bfile.slice)
                        im_data.append(img_temp)

    image_list = []
    mask_list = []
    for it, img in enumerate(im_data):
        temp = utils.preprocess(img,shape,slice_view=slice_view)
        image_list.append(temp[slices[it]])
        mask_list.append(mask[slices[it]])


    return image_list, mask_list