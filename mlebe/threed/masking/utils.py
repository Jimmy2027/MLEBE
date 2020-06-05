import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import scipy.ndimage


def pad_img(img, shape):
    padded = np.empty((img.shape[0], shape[0], shape[1]))

    if img.shape[1] > img.shape[2]:
        for i in range(img.shape[0]):
            padd = img.shape[1] - img.shape[2]
            temp_padded = np.pad(img[i, ...], ((0,0), (padd // 2, img.shape[1] - padd // 2 - img.shape[2])), 'constant')
            padded[i] = cv2.resize(temp_padded, (shape[1], shape[0]))

    elif img.shape[1] < img.shape[2]:
        for i in range(img.shape[0]):
            padd = img.shape[2] - img.shape[1]
            temp_padded = np.pad(img[i, ...], ((padd // 2, img.shape[2] - padd // 2 - img.shape[1]),(0,0)), 'constant')
            padded[i] = cv2.resize(temp_padded, (shape[1], shape[0]))
    else:
        temp = cv2.resize(img[i], (shape[1], shape[0]))
        padded[i] = temp
    return padded




def preprocess(img, shape,slice_view, switched_axis = False):
    """
    - moves axis such that (x,y,z) becomes (z,x,y)
    - transforms the image such that shape is (z,shape). If one dimension is bigger than shape -> downscale, if one dimension is smaller -> zero-pad around the border
    - normalizes the data
    :param img: img with shape (x,y,z)
    :return: img with shape (z,shape)
    """
    if switched_axis == False:
        if slice_view == 'coronal':
            img = np.moveaxis(img, 1, 0)
        elif slice_view == 'axial':
            img = np.moveaxis(img, 2, 0)

    img_data = pad_img(img, shape)
    img_data = data_normalization(img_data)

    return img_data

def data_normalization(data):
    """

    :param data: shape: (y, x)
    :return: normalised input
    """
    data = data*1.
    data = np.clip(data, 0, np.percentile(data, 99))

    data = data - np.amin(data)
    if np.amax(data) != 0:
        data = data / np.amax(data)
    return data



def arrange_mask(img, mask, save_dir = False, visualisation = False):

    new_mask = mask[:,:,:]
    new_mask[img == 0] = 0
    fixed_mask = new_mask[:, :, :]
    structure = [[1,0,1], [1,1,1], [0,1,0]]

    for i in range(new_mask.shape[0]):
        fixed_mask[i] = scipy.ndimage.morphology.binary_fill_holes(new_mask[i], structure=structure)

    if visualisation == True:
        save_datavisualisation([img,mask,new_mask,fixed_mask], save_dir + 'visualisation/arrange_mask/')

    return fixed_mask

def pred_volume_stats(mask_pred, save_path, file_name, model_path):
    import pandas as pd

    unique, counts = np.unique(mask_pred, return_counts=True)
    volume = dict(zip(unique, counts))[1]
    if 'T2' in file_name:
        contrast = 'T2'
    elif 'bold' in file_name:
        contrast = 'BOLD'
    elif 'cbv' in file_name:
        contrast = 'CBV'
    if not os.path.isfile(os.path.join(save_path, 'pred_volume.csv')):
        pred_volume_df = pd.DataFrame(columns=['file_name', 'Contrast', 'Volume', 'model_path'])
    else:
        pred_volume_df = pd.read_csv(os.path.join(save_path, 'pred_volume.csv'))
    pred_volume_df = pred_volume_df.append(pd.DataFrame([[file_name, contrast, volume, model_path]], columns=['file_name', 'Contrast', 'Volume', 'model_path']), sort= False)
    pred_volume_df.to_csv(os.path.join(save_path, 'pred_volume.csv'), index = False)
    return