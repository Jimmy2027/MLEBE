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



def bids_tester(save_path, model, remote, shape, epochs, threshold = 0, test =True):
    if remote == False:
        path = '/Users/Hendrik/Documents/mlebe_data/resampled/'
    else:
        path = '/var/tmp/resampled/'
    save_path = save_path + '/bids_ep{epoch}/'.format(epoch = epochs)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data = []  # list of Nifti1Images
    for o in os.listdir(path):
        if not o.startswith('.'):
            img = nib.load(os.path.join(path, o))
            data.append(img)

    y_pred = []
    img_datas = []

    if test:
        data = data[:5]

    for i in data:
        img_data = i.get_data()
        img_data = utils.preprocess(img_data, shape)
        i = np.expand_dims(img_data, -1)
        temp = model.predict(i, verbose=0)
        if threshold != 0:
            temp = np.where(temp > threshold, 1, 0)
        y_pred.append(temp)
        img_datas.append(img_data)
    print('\nMax y_pred: ', np.max(np.concatenate(y_pred)))
    if np.max(np.concatenate(y_pred)) == 0:
        print('Predictions are zero! Max y_pred: ', np.max(np.concatenate(y_pred)))
        return False

    temp = np.concatenate(y_pred, 0)
    plt.figure()
    plt.hist(np.unique(temp))
    plt.title('Histogram of the pixel values from the predicted masks')
    plt.savefig(os.path.join(save_path, 'hist.png'))

    file_names = []
    for i in range(len(y_pred)):
        x_test_affine = data[i].affine
        x_test_header = data[i].header
        file_name = os.path.basename(data[i].file_map['image'].filename)
        file_names.append(file_name)
        mask_temp = np.moveaxis(y_pred[i], 0, 2)
        img_temp = np.moveaxis(img_datas[i], 0, 2)
        # img = nib.Nifti1Image(temp, x_test_affine, x_test_header)
        mask = nib.Nifti1Image(mask_temp, x_test_affine)
        nib.save(mask, os.path.join(save_path, 'mask_' + file_name))
        img = nib.Nifti1Image(img_temp, x_test_affine)
        nib.save(img, os.path.join(save_path, 'resized_' + file_name))

    output = []
    for i in range(len(data)):
        output.append(np.squeeze(y_pred[i]))

    utils.save_datavisualisation2(img_datas, output, save_path, index_first=True, normalized= True, file_names=file_names)

    return True


if __name__ == '__main__':

    # resample_save_path = '/var/tmp/resampled/'
    # utils.resample_bidsdata(resample_save_path)

    remote = False
    test = True
    loss = 'bincross'
    shape = (128, 128)
    threshold = 0

    if remote == False:
        path = '/Users/Hendrik/Documents/mlebe_data/resampled/'
        model_path = '/Users/Hendrik/Documents/Semester_project/results/unet_ep01_val_loss5.48.hdf5'
        save_path = '/Users/Hendrik/Documents/mlebe_data/temp_bids/'
    else:
        path = '/var/tmp/resampled/'
        model_path = '/home/hendrik/src/mlebe/results/training_results/Dice_50/unet_ep05_val_loss0.04.hdf5'
        save_path = '/home/hendrik/src/mlebe/results/bids_predictions_thr{}/'.format(threshold)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data = []  # list of Nifti1Images
    for o in os.listdir(path):
        if not o.startswith('.'):
            img = nib.load(os.path.join(path, o))
            data.append(img)

    if loss == 'dice':
        model = keras.models.load_model(model_path, custom_objects={'dice_coef_loss': unet.dice_coef_loss})
    elif loss == 'bincross':
        model = keras.models.load_model(model_path)
    else:
        print('wrong loss function defined')


