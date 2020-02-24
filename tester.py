import numpy as np
from tensorflow import keras
import scoring_utils as su
import os
import nibabel as nib
import data_loader as dl
import pickle
from matplotlib import pyplot as plt
import Utils_fd.load_blacklisted as bl
import utils
import unet
# import bids_tester
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

test = False
remote = True
loss = 'dice'
slice_view = 'coronal'

if remote == False:
    path = '/Users/Hendrik/Desktop/new_new_hope3/dice_600_2019-12-18'
    save_dir = '/Users/Hendrik/Documents/mlebe_data/temp/'
    model_dir = path + '/1_Step/unet_ep381_val_loss0.05.hdf5'
    # if test == True:
    #     model_dir = '/Users/Hendrik/Documents/mlebe_data/models/sixty_four/dice/unet_ep01_val_loss0.67.hdf5'

else:
    path = '/mnt/data/mlebe_data/results/func_training/dice_600_2020-02-21'
    save_dir = '/home/hendrik/src/mlebe/tmp/func_test_ftrain/'
    model_dir = path + '/1_Step/model_ep86.h5'
# /home/hendrik/src/mlebe/.scratch/mlebe_data/results/no_tf_lr_64/dice_600_2020-01-30/1_Step/model_ep230.h5
# /mnt/data/mlebe_data/results/no_blacklist/dice_600_2020-01-31/1_Step/model_ep87.h5
# /mnt/data/mlebe_data/results/new_bl128/dice_600_2020-02-07/1_Step/model_ep483.h5
# /mnt/data/mlebe_data/results/new_bl128/dice_600_2020-02-07/1_Step/unet_ep380_val_loss0.04.hdf5

if test == True:
    import shutil

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if loss == 'dice':
    model = keras.models.load_model(model_dir, custom_objects = {'dice_coef_loss': unet.dice_coef_loss})
elif loss == 'bincross':
    model = keras.models.load_model(model_dir)
else: print('wrong loss function defined')

shape = (128, 128)

xfile = open(path + '/x_test_struct.pkl', 'rb')
x_test_struct = pickle.load(xfile)
xfile.close()

yfile = open(path + '/y_test_struct.pkl', 'rb')
y_test_struct = pickle.load(yfile)
yfile.close()

x_test, x_test_affines, x_test_headers, file_names = x_test_struct['x_test'], x_test_struct['x_test_affines'], x_test_struct['x_test_headers'], x_test_struct['file_names']
y_test, y_test_affines, y_test_headers = y_test_struct['y_test'], y_test_struct['y_test_affines'], y_test_struct['y_test_headers']
print('len xtest: ', len(x_test))
print('len ytest: ', len(y_test))
print(x_test[0].shape)

# y_pred = np.load(path +'/y_pred.npy', allow_pickle= True)
#
# x_test = [i[0::10] for i in x_test]
# y_test = [i[0::10] for i in y_test]

if test == True:
    x_test = x_test[:10]
    y_test = y_test[:10]


y_pred = []   #predictions of the test set
dice_scores_string = []
dice_scores = []
dice_scores_thr = []
dice_scores_postpr = []
for hah, i in enumerate(x_test):
    print('img ' + str(hah) + ' of ' + str(len(x_test)))
    img_pred = np.empty((i.shape))
    dice_score_img = []
    dice_score_img_thr = []
    for slice in range(i.shape[0]):
        temp = np.expand_dims(i[slice], -1)  # expand dims for channel
        temp = np.expand_dims(temp, 0)  # expand dims for batch
        prediction = model.predict(temp, verbose=0)
        prediction = np.squeeze(prediction)
        prediction_postpr = utils.remove_outliers(prediction, visualisation = True, save_dir = save_dir + 'remove_outliers/')
        dice_scores_postpr.append(su.dice(y_test[hah][slice], np.where(prediction_postpr > 0.9, 1, 0)))
        img_pred[slice, ...] = prediction
        dice_scores.append(su.dice(y_test[hah][slice], np.where(prediction > 0.9, 1, 0)))
        dice_score_img.append('dice: ' + str(np.round(su.dice(y_test[hah][slice], np.where(prediction > 0.9, 1, 0)), 3)))
        dice_score_img_thr.append('dice: ' + str(np.round(su.dice(y_test[hah][slice], np.where(prediction > 0.5, 1, 0)), 3)))

    y_pred.append(img_pred)
    dice_scores_string.append(dice_score_img)
    dice_scores_thr.append(dice_score_img_thr)
print('done predicting on test set')
# temp = np.concatenate(y_pred, 0)
median_dice_score = np.median(dice_scores)
dice_score = np.average(dice_scores)

thresholds = [0.9, 0.5]
outputs = []
slice_titles = [None, None, dice_scores_string, None]
row_titles = ['x_test']

corr_temp = [utils.corr(y, x) for x, y in zip(x_test, y_test)]

row_titles.append('y_test' + '\n corr: ' + str(np.round(np.average(corr_temp), 3)))
correlations_thr = []

for thr in thresholds:
    thr_dice = []
    if thr == 0:
        outputs.append([np.squeeze(img) for img in y_pred])
        dice_temp = [su.dice(np.squeeze(img), y_true) for img, y_true in zip(y_pred, y_test)]
        corr_temp = [utils.corr(img, x) for img, x in zip(y_pred, x_test)]
        row_titles.append('Prediction \n' + 'Dice: ' + str(np.round(np.average(dice_temp), 3)) + '\n corr: ' + str(np.round(np.average(corr_temp),3)))
    else:
        outputs.append([np.where(np.squeeze(img) > thr, 1, 0) for img in y_pred])
        dice_temp = [su.dice(np.where(np.squeeze(img) > thr, 1, 0), y_true) for img, y_true in zip(y_pred, y_test)]
        corr_temp = [utils.corr(np.where(img > thr, 1, 0), x) for img, x in zip(y_pred, x_test)]
        row_titles.append('thr: ' + str(thr) + '\n ' + 'Dice: ' + str(np.round(np.average(dice_temp), 3)) + '\n corr: ' + str(np.round(np.average(corr_temp),3)))

utils.compute_correlation(np.concatenate(x_test), np.concatenate(y_test), np.concatenate(outputs[0]), save_dir)
list = [x_test, y_test]
for o in outputs:
    list.append(o)
slice_titles[-1] = dice_scores_thr

utils.save_datavisualisation_plt(list, save_dir, normalized=True, file_names=file_names, slice_titles=slice_titles, row_titles=row_titles)

dice_score = np.average(dice_scores)
textfile = open(save_dir + 'dice_score.txt', 'w+')
textfile.write('mean dice_postpr score: ' + str(np.mean(dice_scores_postpr))+ '\n\n')
textfile.write('len xtest: ' +  str(len(x_test)) + '\n\n')
textfile.write('std: ' + str(np.std(dice_scores)) + '\n\n')
textfile.write('average dice score ' + str(dice_score) + '\n\n')
textfile.write('median dice score: ' + str(median_dice_score) + '\n\n')
textfile.close()
print('median Dice score: ', median_dice_score)
print('average Dice score: ', dice_score)
print('std: ', np.std(dice_scores))
file_namess = []
for i in range(len(y_pred)):
    y_test_affine = y_test_affines[i]
    y_test_header = y_test_headers[i]
    file_name = os.path.basename(os.path.basename(file_names[i]))
    file_namess.append(file_name)
    temp = np.moveaxis(y_pred[i], 0, 2)
    img = nib.Nifti1Image(y_pred[i], y_test_affine, y_test_header)
    nib.save(img, os.path.join(save_dir, 'mask_' + file_name))

np.save(save_dir + 'y_pred_{}dice'.format(np.round(dice_score, 4)), y_pred)
remote = 'hongg'
# bids_tester.bids_tester(save_dir, model, remote, shape, slice_view=slice_view, test = False)
