import numpy as np
from tensorflow import keras
import scoring_utils as su
import os
import nibabel as nib
import data_loader as dl
import pickle
from matplotlib import pyplot as plt

import utils
import unet
import bids_tester


test = True
remote = False
loss = 'dice'

if remote == False:
    path = '/Users/Hendrik/Desktop/new_hope0/training_results/dice_1200_2019-12-09'
    save_dir = '/Users/Hendrik/Documents/mlebe_data/temp/'
    model_dir = path + '/1_Step/unet_ep70_val_loss0.01.hdf5'
    print(model_dir)
else:
    path = '/home/hendrik/src/mlebe/results/training_results/Dice_50'
    save_dir = '/home/hendrik/src/mlebe/tmp/'
    model_dir = path + '/unet_ep70_val_loss0.01.hdf5'

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

# y_pred = np.load(path +'/y_pred.npy', allow_pickle= True)

if test == True:
    x_test = x_test[:10]
    y_test = y_test[:10]


y_pred = []
dice_scores = []

for i in x_test:
    i = np.expand_dims(i, -1)
    temp = model.predict(i, verbose=1)
    y_pred.append(temp)
    dice_scores.append(su.dice(i, temp))

temp = np.concatenate(y_pred, 0)
plt.figure()
plt.hist(np.squeeze(temp).flatten(), bins = 'auto')
plt.yscale('log')
plt.title('Histogram of the pixel values from the predicted masks')
plt.savefig(os.path.join(save_dir, 'hist.png'))
plt.close()

# _ = plt.hist(np.squeeze(temp).flatten(), bins='auto')
# plt.savefig(os.path.join(save_dir, 'hist.png'))


thresholds = [0, 0.5, 0.7, 0.8, 0.9]
outputs = []
for thr in thresholds:
    if thr == 0:
        outputs.append([np.squeeze(img) for img in y_pred])
    else:
        outputs.append([np.where(np.squeeze(img) > thr, 1, 0) for img in y_pred])
list = [x_test, y_test]
for o in outputs:
    list.append(o)
utils.save_datavisualisation(list, save_dir, file_name_header='thr[0,0.5,0.7,0.8,0.9]', normalized=True, file_names=file_names)

dice_score = np.median(dice_scores)
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

bids_tester.bids_tester(save_dir, model, remote, shape)
