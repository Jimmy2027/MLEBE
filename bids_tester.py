import os
import utils
import nibabel as nib
import numpy as np
import utils
from tensorflow import keras
from matplotlib import pyplot as plt

import data_loader as dl

# resample_save_path = '/var/tmp/resampled/'
# utils.resample_bidsdata(resample_save_path)

local = False
test = True
shape = (128, 128)
threshold = 0.5

if local == True:
    path = '/Users/Hendrik/Documents/mlebe_data/resampled/'
    model_path = '/Users/Hendrik/Documents/Semester_project/results/unet_ep01_val_loss5.48.hdf5'
    save_path = '/Users/Hendrik/Documents/mlebe_data/temp_bids/'
else:
    path = '/var/tmp/resampled/'
    model_path = '/home/hendrik/src/mlebe/results/training_results/Dice_50/unet_ep05_val_loss0.04.hdf5'
    save_path = '/home/hendrik/src/mlebe/results/bids_predictions_thr{}/'.format(threshold)


if not os.path.exists(save_path):
    os.makedirs(save_path)
data = []   #list of Nifti1Images
for o in os.listdir(path):
    if not o.startswith('.'):
        print(o)
        img = nib.load(os.path.join(path, o))
        data.append(img)

model = keras.models.load_model(model_path)

y_pred = []
img_datas = []

if test:
    data = data[:5]

for i in data:
    img_data = i.get_data()
    img_data = utils.preprocess(img_data, shape)
    i = np.expand_dims(img_data, -1)
    temp = model.predict(i, verbose=1)
    temp = np.where(temp > threshold, 1, 0)
    y_pred.append(temp)
    img_datas.append(img_data)

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




