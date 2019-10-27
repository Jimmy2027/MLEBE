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
    path = '/Users/Hendrik/Desktop/resampled/'
    model_path = '/Users/Hendrik/Desktop/results/unet_ep35_val_loss0.07.hdf5'
else:
    path = '/var/tmp/resampled/'
    model_path = '/src/mlebe/results/unet_ep35_val_loss0.07.hdf5'


data = []
file_names = []
for o in os.listdir(path):
    if not o.startswith('.'):
        file_names.append(o)
        img = nib.load(os.path.join(path, o))
        img_data = img.get_data()
        temp = np.moveaxis(img_data, 0, 1)
        img_data = utils.pad_img(temp)
        img_data = utils.data_normalization(img_data)
        data.append(img_data)

model = keras.models.load_model(model_path)

y_pred = []
for i in data:
    i = np.expand_dims(i, -1)
    y_pred.append(model.predict(i, verbose=1))

for i in range(len(y_pred)):
    file_name = file_names[i]
    nib.save(y_pred, os.path.join('results', 'mask_'+file_name))
output = []
for i in range(len(data)):
    output.append(np.squeeze(y_pred[i]))

utils.save_datavisualisation2(data, y_pred, '/Users/Hendrik/Desktop/results/', index_first=True, normalized= True)

something = 0


