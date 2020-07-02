import os
from os import path

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt

"""
This script plot all the images from the dataset for visualisation. 
This can be used to create a blacklist.

"""

dir = path.abspath(path.expanduser('/mnt/data/mlebe_data/'))

mask_dir = '/usr/share/mouse-brain-atlases/'

save_dir = 'Unpreprocessed/'

im_data = []
for o in os.listdir(mask_dir):
    if o == 'dsurqec_200micron_mask.nii':
        im_data.append(os.path.join(mask_dir, o))


im_data = np.sort(im_data)


for i in im_data:
    img = nib.load(i)
    mask = img.get_data()
    mask = np.moveaxis(mask, 1, 0)

for o in os.listdir(dir):
    if o != 'irsabi':
        for x in os.listdir(os.path.join(dir, o)):
            if x.endswith('preprocessing'):
                for root, dirs, files in os.walk(os.path.join(dir, o, x)):
                    for file in files:
                        if file.endswith("_T2w.nii.gz"):
                            path = root.split('/')
                            path_sep = path[-5:]
                            path = path_sep[0]
                            for elem in path_sep[1:]:
                                path = os.path.join(path, elem)
                            print(path)
                            os.makedirs(save_dir + path, exist_ok = True)
                            img = nib.load(os.path.join(root, file))
                            img_data = img.get_data()
                            img_data = np.moveaxis(img_data, 1, 0)

                            for it in range(img_data.shape[0]):
                                plt.imshow(np.squeeze(img_data[it, ...]), cmap='gray')
                                plt.imshow(np.squeeze(mask[it, ...]), alpha=0.6, cmap='Blues')
                                plt.axis('off')
                                plt.savefig(save_dir + path +  '/{name}_{it}.pdf'.format(name = os.path.basename(file),it=it), format='pdf')
                                plt.close()


