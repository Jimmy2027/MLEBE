import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""   #predict on CPU instead of GPU
import utils
import nibabel as nib
import numpy as np
import utils
import unet
from tensorflow import keras
from matplotlib import pyplot as plt
from predict_mask import predict_mask

import data_loader as dl



def bids_tester(save_path, model, remote, shape, slice_view, epochs = 0, test =True, threshold = 0.9):
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
    if remote == 'local':
        path = '/Users/Hendrik/Documents/mlebe_data/resampled/'
    elif remote == 'hongg':
        path = '/home/hendrik/src/mlebe/resampled/'
        if not os.path.exists(path):
            utils.resample_bidsdata(path)
    elif remote == 'leonhard':
        path = '/cluster/scratch/klugh/resampled/'
        if not os.path.exists(path):
            utils.resample_bidsdata(path)
    elif remote == 'epfl':
        path ='/home/klug/Hendrik/MLEBE/resampled'
        if not os.path.exists(path):
            utils.resample_bidsdata(path)
    save_path = save_path + 'bids_ep{epoch}/'.format(epoch = epochs)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data = []  # list of Nifti1Images
    for o in os.listdir(path):
        if not o.startswith('.'):
            img = nib.load(os.path.join(path, o))
            data.append(img)

    y_pred = []
    y_pred_thr = []
    img_datas = []

    if test:
        data = data[:5]

    for i in data:
        img_data = i.get_data()
        img_data = utils.preprocess(img_data, shape, slice_view=slice_view)
        input = np.expand_dims(img_data, -1)
        temp = model.predict(input, verbose=0)
        temp_thr = np.where(np.squeeze(temp) > threshold, 1, 0)
        y_pred.append(temp)
        y_pred_thr.append(temp_thr)
        img_datas.append(img_data)
        # if not os.path.exists(save_path + '{a}/'.format(a = os.path.basename(i.file_map['image'].filename))):
        #     os.makedirs(save_path + '{a}/'.format(a = os.path.basename(i.file_map['image'].filename)))
        # for j, slice in enumerate(img_data):
        #     plt.imshow(slice, cmap='gray')
        #     plt.imshow(temp_thr[j], alpha=0.6, cmap='Blues')
        #     if not os.path.exists(save_path):
        #         os.makedirs(save_path)
        #     plt.savefig(save_path + '{a}/slice{b}.pdf'.format(a = os.path.basename(i.file_map['image'].filename), b=j), format="pdf", dpi=300)
        #     plt.close()
    print('\nMax y_pred: ', np.max(np.concatenate(y_pred)))
    if np.max(np.concatenate(y_pred)) == 0:
        print('Predictions are zero! Max y_pred: ', np.max(np.concatenate(y_pred)))
        return False

        temp = np.concatenate(y_pred, 0)
    plt.figure()
    plt.hist(np.unique(temp))
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

    thresholds = [0, 0.9]
    outputs = []
    for thr in thresholds:
        if thr == 0:
            outputs.append([np.squeeze(img) for img in y_pred])
        else:
            outputs.append([np.where(np.squeeze(img) > thr, 1, 0) for img in y_pred])
    list = [img_datas]
    for o in outputs:
        list.append(o)

    utils.save_datavisualisation(list, save_path,file_name_header='thr[0,0.9]', normalized=True, file_names=file_names)


    return True


if __name__ == '__main__':

    # resample_save_path = '/var/tmp/resampled/'
    # utils.resample_bidsdata(resample_save_path)

    remote = 'local'
    test = True
    loss = 'dice'
    shape = (64, 64)
    threshold = 0

    if remote == 'local':
        path = '/Users/Hendrik/Documents/mlebe_data/resampled/'
        model_path = '/Users/Hendrik/Desktop/new_new_hope3/dice_600_2019-12-18/1_Step/unet_ep381_val_loss0.05.hdf5'
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

    import shutil
    save_dir = 'temp_s/'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    os.mkdir(save_dir)

    something = nib.load('/Users/Hendrik/Desktop/sub-4011_ses-ofMcF2_acq-TurboRARElowcov_T2w_corrected.nii.gz')
    predict_mask(something, save_dir)


    save_dir = 'temp_bids/'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    predict_mask(data[0], save_dir)



    #
    #
    # if loss == 'dice':
    #     model = keras.models.load_model(model_path, custom_objects={'dice_coef_loss': unet.dice_coef_loss})
    # elif loss == 'bincross':
    #     model = keras.models.load_model(model_path)
    # else:
    #     print('wrong loss function defined')
    #
    # bids_tester(save_path, model, remote, shape, 'coronal')
