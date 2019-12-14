import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""   #predict on CPU instead of GPU
import utils
import nibabel as nib
import numpy as np
import utils
import unet
from tensorflow import keras
from matplotlib import pyplot as plt

import data_loader as dl



def func_tester(remote, slice_view, test, model, shape):
    """
    Preprocesses the unpreprocessed bidsdata and predicts a mask for it

    :param save_path:
    :param model:
    :param remote:
    :param shape:
    :param epochs:
    :param threshold: Threshold of the mask (only used for the nib images)
    :param test:
    :return: True if predictions are greater than 0, else False
    """

    if remote == 'höngg':
        image_dir_remote = '/mnt/scratch/'
        data_dir = '/usr/share/mouse-brain-atlases/'
        img_datas = dl.load_func_img(image_dir_remote, test)
    elif remote == 'leonhard':
        image_dir_remote = '/cluster/scratch/klugh/preprocessed'
        data_dir = '/cluster/scratch/klugh/mouse-brain-atlases/'
        img_datas = dl.load_func_img(image_dir_remote, test)

    else:
        image_dir = '/Users/Hendrik/Documents/mlebe_data/preprocessed'
        img_datas = dl.load_func_img(image_dir, test)

        data_dir = '/Users/Hendrik/Documents/mlebe_data/mouse-brain-atlases/'  # local

    data = []
    img_data = []
    for i in img_datas:
        temp = i.get_data()
        data.append(temp)
        if slice_view == 'coronal':
            img_temp = np.empty((temp.shape[1], shape[0],shape[1], temp.shape[-1]))
        elif slice_view == 'sagittal':
            img_temp = np.empty((temp.shape[0], shape[0], shape[1], temp.shape[-1]))
        elif slice_view == 'axial':
            img_temp = np.empty((temp.shape[2], shape[0], shape[1], temp.shape[-1]))
        for frame in range(img_temp.shape[-1]):
            img_temp[...,frame] = utils.preprocess(temp[...,frame], shape, slice_view)

        img_data.append(img_temp)


    save_path = 'func_tester/'


    if not os.path.exists(save_path):
        os.makedirs(save_path)


    y_pred = []
    y_pred_thr = []
    img_datas = []

    threshold = 0


    for j in img_data:
        placeholder = np.empty((j.shape))
        for i in range(j.shape[-1]):

            temp = model.predict(j[...,i], verbose=0)
            temp = np.expand_dims(temp, -1)
            if threshold != 0:
                temp_thr = np.where(temp > threshold, 1, 0)
            placeholder[...,i] = temp
        y_pred.append(placeholder)


    something = 0
    temp = np.concatenate(y_pred, 0)
    plt.figure()
    plt.hist(np.squeeze(temp).flatten(), bins='auto')
    plt.yscale('log')
    plt.title('Histogram of the pixel values from the predicted masks')
    plt.savefig(os.path.join(save_path, 'hist.png'))
    plt.close()


    file_names = []
    for i in range(len(y_pred)):
        x_test_affine = data[i].affine
        x_test_header = data[i].header
        file_name = os.path.basename(data[i].file_map['image'].filename)
        file_names.append(file_name)
        if slice_view == 'coronal':
            img_temp = np.moveaxis(img_datas[i], 1, 0)
            mask_temp = np.moveaxis(y_pred_thr[i], 1, 0)
        elif slice_view == 'axial':
            mask_temp = np.moveaxis(y_pred_thr[i], 0, 2)
            img_temp = np.moveaxis(img_datas[i], 0, 2)
        else:
            mask_temp = y_pred_thr[i]
            img_temp = img_datas[i]

        # img = nib.Nifti1Image(temp, x_test_affine, x_test_header)
        mask = nib.Nifti1Image(mask_temp, x_test_affine)
        nib.save(mask, os.path.join(save_path, 'mask_thr{}_'.format(threshold) + file_name))
        img = nib.Nifti1Image(img_temp, x_test_affine)
        nib.save(img, os.path.join(save_path, 'resized_thr{}'.format(threshold) + file_name))

    thresholds = [0, 0.5, 0.7, 0.8, 0.9]
    outputs = []
    for thr in thresholds:
        if thr == 0:
            outputs.append([np.squeeze(img) for img in y_pred])
        else:
            outputs.append([np.where(np.squeeze(img) > thr, 1, 0) for img in y_pred])
    list = [img_datas]
    for o in outputs:
        list.append(o)

    utils.save_datavisualisation(list, save_path,file_name_header='thr[0,0.5,0.7,0.8,0.9]', normalized=True, file_names=file_names)


    return True


if __name__ == '__main__':

    # resample_save_path = '/var/tmp/resampled/'
    # utils.resample_bidsdata(resample_save_path)



    remote = False
    test = True
    slice_view = 'coronal'
    loss = 'dice_bincross'
    shape = (64, 64)
    threshold = 0

    if remote == False:
        path = '/Users/Hendrik/Documents/mlebe_data/resampled/'
        model_path = '/Users/Hendrik/Documents/mlebe_data/models/dice_bincross/unet_ep01_val_loss1.55.hdf5'
        save_path = '/Users/Hendrik/Documents/mlebe_data/temp_bids/'
    else:
        path = '/var/tmp/resampled/'
        model_path = '/home/hendrik/src/mlebe/results/training_results/Dice_50/unet_ep05_val_loss0.04.hdf5'
        save_path = '/home/hendrik/src/mlebe/results/bids_predictions_thr{}/'.format(threshold)

    if loss == 'dice':
        model = keras.models.load_model(model_path, custom_objects={'dice_coef_loss': unet.dice_coef_loss})
    elif loss == 'bincross':
        model = keras.models.load_model(model_path)
    elif loss == 'dice_bincross':
        model = keras.models.load_model(model_path, custom_objects={'dice_bincross_loss': unet.dice_bincross_loss})
    else:
        print('wrong loss function defined')

    func_tester(remote, slice_view, test, model, shape)


