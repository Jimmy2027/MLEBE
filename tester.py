import numpy as np
from tensorflow import keras
import utils
import unet
import scoring_utils as su
import os
import nibabel as nib
import data_loader as dl
import pickle

remote = False

if remote == False:
    path = '/Users/Hendrik/Documents/mlebe_data/results/test'
    save_dir = '/Users/Hendrik/Documents/mlebe_data/temp/'
    model_dir = path + '/unet_ep01_val_loss5.35.hdf5'
else:
    path = 'home/hendrik/src/mlebe/results/training_results/Dice_50'
    save_dir = '/home/hendrik/src/mlebe/tmp/'
    model_dir = path + '/unet_ep05_val_loss0.04.hdf5'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model = keras.models.load_model(model_dir, custom_objects = {'dice_coef_loss': unet.dice_coef_loss})

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

y_pred = []
dice_scores = []

for i in x_test:
    i = np.expand_dims(i, -1)
    temp = model.predict(i, verbose=1)
    y_pred.append(temp)
    dice_scores.append(su.dice(i, temp))

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


output = []

for i in range(len(y_test)):
    output.append(np.squeeze(y_pred[i]))

utils.save_datavisualisation3(x_test, y_test, output, save_dir, index_first=True, normalized= True, file_names = file_namess)

np.save(save_dir + 'y_pred_{}dice'.format(np.round(dice_score, 4)), y_pred)