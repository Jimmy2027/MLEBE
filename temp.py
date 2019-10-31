import numpy as np
import os
import utils
import nibabel as nib
import numpy as np
import utils
from tensorflow import keras

data = np.load('/Users/Hendrik/Documents/Semester_project/temp/data.npy')
shape = (64, 128)
local = True


if local == True:
    path = '/Users/Hendrik/Documents/mlebe_data/resampled/'
    model_path = '/Users/Hendrik/Documents/Semester_project/results/unet_ep01_val_loss5.48.hdf5'
    save_path = '/Users/Hendrik/Documents/mlebe_data/temp_bids/'
else:
    path = '/var/tmp/resampled/'
    model_path = '/home/hendrik/src/mlebe/results1/unet_ep35_val_loss0.07.hdf5'
    save_path = '/home/hendrik/src/mlebe/results/bids_predictions/'

model = keras.models.load_model(model_path)

y_pred = []
img_datas = []
data = data[:5]
for i in data:
    img_data = i.get_data()
    img_data = utils.preprocess(img_data, shape)
    i = np.expand_dims(img_data, -1)
    y_pred.append(model.predict(i, verbose=1))
    img_datas.append(img_data)

for i in range(len(y_pred)):
    x_test_affine = data[i].affine
    x_test_header = data[i].header
    file_name = os.path.basename(data[i].file_map['image'].filename)
    temp = np.moveaxis(y_pred[i], 0, 2)
    img = nib.Nifti1Image(temp, x_test_affine, x_test_header)
    nib.save(img, os.path.join(save_path, 'mask_' + file_name))

output = []
for i in range(len(data)):
    output.append(np.squeeze(y_pred[i]))

utils.save_datavisualisation2(img_datas, img_datas, save_path, index_first=True, normalized= True)