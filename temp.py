import numpy as np
import os
import utils
import nibabel as nib
import numpy as np
import utils
from tensorflow import keras

data = np.load('/Users/Hendrik/Documents/Semester_project/temp/data.npy')
file_names = np.load('/Users/Hendrik/Documents/Semester_project/temp/file_names.npy')

path = '/Users/Hendrik/Documents/mlebe_data/resampled/'
model_path = '/Users/Hendrik/Documents/Semester_project/results/unet_ep01_val_loss5.48.hdf5'

model = keras.models.load_model(model_path)

y_pred = []
for i in data:
    i = np.expand_dims(i, -1)
    y_pred.append(model.predict(i, verbose=1))

for i in range(len(y_pred)):
    file_name = file_names[i]
    proxy = nib.load()
    img = nib.Nifti1Image(y_pred[i], affine)

    # nib.save(y_pred[i], os.path.join('results', 'mask_'+file_name))
    y_pred[i].to_filename(os.path.join('results', 'mask_'+file_name))

output = []
for i in range(len(data)):
    output.append(np.squeeze(y_pred[i]))

# utils.save_datavisualisation2(data, y_pred, '/Users/Hendrik/Documents/Semester_project/results/', index_first=True, normalized= True)
utils.save_datavisualisation3(data, output, output, '/Users/Hendrik/Documents/Semester_project/results/', index_first=True, normalized= True)
