import os
import utils
import nibabel as nib
import numpy as np
import utils
from tensorflow import keras

import data_loader as dl

# utils.resample_bidsdata()
local = True

if local == True:
    path = '/Users/Hendrik/Documents/mlebe_data/resampled/'
    model_path = '/Users/Hendrik/Documents/Semester_project/results/unet_ep01_val_loss5.48.hdf5'
    save_path = '/Users/Hendrik/Documents/mlebe_data/temp_bids/'
else:
    path = '/var/tmp/resampled/'
    model_path = '/home/hendrik/src/mlebe/results/unet_ep35_val_loss0.07.hdf5'
    save_path = '/home/hendrik/src/mlebe/results/bids_predictions/'


if not os.path.exists(save_path):
    os.makedirs(save_path)
data = []
file_names = []
affines = []
for o in os.listdir(path):
    if not o.startswith('.'):
        print(o)
        file_names.append(o)
        img = nib.load(os.path.join(path, o))
        affines.append(img.affine)
        img_data = img.get_data()
        temp = np.moveaxis(img_data, 0, 1)
        img_data = utils.pad_img(temp)
        img_data = utils.data_normalization(img_data)
        data.append(img_data)

# np.save('/Users/Hendrik/Documents/Semester_project/temp/data',data)
# np.save('/Users/Hendrik/Documents/Semester_project/temp/file_names', file_names)
# np.save('/Users/Hendrik/Documents/Semester_project/temp/affines', affines)
model = keras.models.load_model(model_path)

y_pred = []
for i in data:
    i = np.expand_dims(i, -1)
    y_pred.append(model.predict(i, verbose=1))

for i in range(len(y_pred)):
    file_name = file_names[i]
    temp = np.moveaxis(img_data, 0, 1)
    img = nib.Nifti1Image(temp, affines[i])
    nib.save(img, os.path.join(save_path, 'mask_' + file_name))
output = []
for i in range(len(data)):
    output.append(np.squeeze(y_pred[i]))

utils.save_datavisualisation2(data, output, save_path, index_first=True, normalized= True)




