import numpy as np
from tensorflow import keras
import utils
import os
import nibabel as nib
import data_loader as dl
import pickle

remote = False

if remote == False:
    path = '/Users/Hendrik/Documents/mlebe_data/results/test'
    save_dir = '/Users/Hendrik/Documents/mlebe_data/temp'
else:
    path = '/home/hendrik/src/mlebe/results/50epochs_training_results '
    save_dir = '/home/hendrik/src/mlebe/tmp'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# model = keras.models.load_model('/home/hendrik/src/mlebe/results/unet_ep12_val_loss0.06.hdf5')
# model = keras.models.load_model(path + '/unet_ep01_val_loss16.07.hdf5')

shape = (128, 128)

xfile = open(path + '/x_test_struct.pkl', 'rb')
x_test_struct = pickle.load(xfile)
xfile.close()

yfile = open(path + '/y_test_struct.pkl', 'rb')
y_test_struct = pickle.load(yfile)
yfile.close()

x_test, x_test_affines, x_test_headers, file_names = x_test_struct['x_test'], x_test_struct['x_test_affines'], x_test_struct['x_test_headers'], x_test_struct['file_names']
y_test, y_test_affines, y_test_headers = y_test_struct['y_test'], y_test_struct['y_test_affines'], y_test_struct['y_test_headers']

y_pred = np.load(path +'/y_pred.npy', allow_pickle= True)

# y_pred = []
# for i in x_test:
#     i = np.expand_dims(i, -1)
#     y_pred.append(model.predict(i, verbose=1))


for i in range(len(y_pred)):
    x_test_affine = x_test_affines[i]
    x_test_header = x_test_headers[i]
    file_name = os.path.basename(os.path.basename(file_names[i]))
    file_names.append(file_name)
    # temp = np.moveaxis(y_pred[i], 0, 2)
    img = nib.Nifti1Image(y_pred[i], x_test_affine, x_test_header)
    nib.save(img, os.path.join(save_dir, 'mask_' + file_name))


# output = []
#
# for i in range(len(y_test)):
#     output.append(np.squeeze(y_pred[i]))
#
# utils.save_datavisualisation3(x_test, y_test, output, 'results/', index_first=True, normalized= False, file_names = file_names)
#
