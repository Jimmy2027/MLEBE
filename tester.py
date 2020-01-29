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
import bids_tester
import cv2


test = True
remote = False
loss = 'dice'
slice_view = 'coronal'

if remote == False:
    path = '/Users/Hendrik/Desktop/new_new_hope3/dice_600_2019-12-18'
    save_dir = '/Users/Hendrik/Documents/mlebe_data/temp/'
    model_dir = path + '/1_Step/unet_ep381_val_loss0.05.hdf5'
    # if test == True:
    #     model_dir = '/Users/Hendrik/Documents/mlebe_data/models/sixty_four/dice/unet_ep01_val_loss0.67.hdf5'

else:
    path = '/home/hendrik/src/mlebe/old_results/new_new_hope3/dice_600_2019-12-18'
    save_dir = '/home/hendrik/src/mlebe/tmp/'
    model_dir = path + '/1_Step/unet_ep381_val_loss0.05.hdf5'

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

shape = (64, 64)

"""
Predict on blacklisted
"""

blackl_images, blackl_masks = bl.load_blacklisted(remote, slice_view, shape, save_dir)
blackl_images = blackl_images[:10]
blackl_masks = blackl_masks[:10]

y_pred_bl = []   #predictions of the blacklisted slices

for slice in blackl_images:
    temp = np.expand_dims(slice, -1)
    temp = np.expand_dims(temp, 0)
    prediction = model.predict(temp, verbose = 0)
    y_pred_bl.append(np.squeeze(prediction))

if not os.path.exists(save_dir + 'bl_images/'):
    os.makedirs(save_dir+ 'bl_images/')

utils.compute_correlation(np.concatenate(blackl_images), np.concatenate(blackl_masks), np.concatenate(y_pred_bl), save_dir + 'bl_images/bl_')

for i, img in enumerate(blackl_images):
    plt.imshow(img, cmap='gray')
    plt.imshow(blackl_masks[i], alpha=0.6, cmap='Blues')
    plt.savefig(save_dir + 'bl_images/bl_mask{}.pdf'.format(i), format = "pdf", dpi = 300)
    plt.close()

    plt.imshow(img, cmap='gray')
    plt.imshow(y_pred_bl[i], alpha=0.6, cmap='Blues')
    plt.savefig(save_dir + 'bl_images/bl_pred{}.pdf'.format(i), format = "pdf", dpi = 300)
    plt.close()

xfile = open(path + '/x_test_struct.pkl', 'rb')
x_test_struct = pickle.load(xfile)
xfile.close()

yfile = open(path + '/y_test_struct.pkl', 'rb')
y_test_struct = pickle.load(yfile)
yfile.close()

x_test, x_test_affines, x_test_headers, file_names = x_test_struct['x_test'], x_test_struct['x_test_affines'], x_test_struct['x_test_headers'], x_test_struct['file_names']
y_test, y_test_affines, y_test_headers = y_test_struct['y_test'], y_test_struct['y_test_affines'], y_test_struct['y_test_headers']

# y_pred = np.load(path +'/y_pred.npy', allow_pickle= True)

x_test = [i[0::10] for i in x_test]
y_test = [i[0::10] for i in y_test]

if test == True:
    x_test = x_test[:10]
    y_test = y_test[:10]


y_pred = []   #predictions of the test set
dice_scores_string = []
dice_scores = []
dice_scores_thr = []
for i in y_test:
    img_pred = np.empty((i.shape))
    dice_score_img = []
    dice_score_img_thr = []
    for slice in range(i.shape[0]):
        temp = np.expand_dims(i[slice], -1)  # expand dims for channel
        temp = np.expand_dims(temp, 0)  # expand dims for batch
        prediction = model.predict(temp, verbose=0)
        prediction = np.squeeze(prediction)
        img_pred[slice, ...] = prediction
        dice_scores.append(su.dice(i[slice], np.where(prediction > 0.9, 1, 0)))
        dice_score_img.append('dice: ' + str(np.round(su.dice(i[slice], np.where(prediction > 0.9, 1, 0)), 3)))
        dice_score_img_thr.append('dice: ' + str(np.round(su.dice(i[slice], np.where(prediction > 0.5, 1, 0)), 3)))

    y_pred.append(img_pred)
    dice_scores_string.append(dice_score_img)
    dice_scores_thr.append(dice_score_img_thr)

temp = np.concatenate(y_pred, 0)
dice_score = np.median(dice_scores)

plt.figure()
plt.hist(np.squeeze(temp).flatten(), bins = 'auto')
plt.yscale('log')
plt.title('Histogram of the pixel values from the predicted masks')
plt.savefig(os.path.join(save_dir, 'hist.png'))
plt.close()

# _ = plt.hist(np.squeeze(temp).flatten(), bins='auto')
# plt.savefig(os.path.join(save_dir, 'hist.png'))



thresholds = [0.9, 0.5]
outputs = []
slice_titles = [None, None, dice_scores_string, None]
row_titles = ['x_test']

corr_temp = [utils.corr(y, x) for x, y in zip(x_test, y_test)]

row_titles.append('y_test' + '\n corr: ' + str(np.round(np.median(corr_temp), 3)))
correlations_thr = []

for thr in thresholds:
    thr_dice = []
    if thr == 0:
        outputs.append([np.squeeze(img) for img in y_pred])
        dice_temp = [su.dice(np.squeeze(img), y_true) for img, y_true in zip(y_pred, y_test)]
        # corr_temp = [np.correlate(np.squeeze(img), x) for img, x in zip(y_pred, x_test)]
        corr_temp = [utils.corr(img, x) for img, x in zip(y_pred, x_test)]
        row_titles.append('Prediction \n' + 'Dice: ' + str(np.round(np.median(dice_temp), 3)) + '\n corr: ' + str(np.round(np.median(corr_temp),3)))
    else:
        outputs.append([np.where(np.squeeze(img) > thr, 1, 0) for img in y_pred])
        dice_temp = [su.dice(np.where(np.squeeze(img) > thr, 1, 0), y_true) for img, y_true in zip(y_pred, y_test)]
        # corr_temp = [np.correlate(np.where(np.squeeze(img) > thr, 1, 0), x) for img, x in zip(y_pred, x_test)]
        corr_temp = [utils.corr(np.where(img > thr, 1, 0), x) for img, x in zip(y_pred, x_test)]
        row_titles.append('thr: ' + str(thr) + '\n ' + 'Dice: ' + str(np.round(np.median(dice_temp), 3)) + '\n corr: ' + str(np.round(np.median(corr_temp),3)))

utils.compute_correlation(np.concatenate(x_test), np.concatenate(y_test), np.concatenate(outputs[0]), save_dir)
list = [x_test, y_test]
for o in outputs:
    list.append(o)
slice_titles[-1] = dice_scores_thr


# list_new = []
# for l in list:
#     new_elem = []
#     for elem in l:
#         new_i = np.empty((elem.shape[0],elem.shape[1]*8,elem.shape[2] *8))
#         for j,i in enumerate(elem):
#             try:
#                 new_i[j] = cv2.resize(elem[j], (elem[j].shape[0]*8, elem[j].shape[1]*8))
#             except Exception as e:
#                 print(e)
#         new_elem.append(new_i)
#     list_new.append(new_elem)
#         # elem = elem[i = cv2.resize(i, (i.shape[0]*2, i.shape[1]*2)) for i in elem]
utils.save_datavisualisation_plt(list, save_dir, normalized=True, file_names=file_names, slice_titles=slice_titles, row_titles=row_titles)

dice_score = np.median(dice_scores)
textfile = open(save_dir + 'dice_score.txt', 'w+')
textfile.write(str(dice_score))
textfile.close()
print('median Dice score: ', dice_score)
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

# bids_tester.bids_tester(save_dir, model, remote, shape, slice_view=slice_view)
