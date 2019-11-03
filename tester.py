import numpy as np
from tensorflow import keras
import utils
import os
import nibabel as nib
import data_loader as dl

path = '/Users/Hendrik/Documents/mlebe_data/results/'
save_dir = '/Users/Hendrik/Documents/mlebe_data/temp'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

model = keras.models.load_model(path + 'unet_ep10_val_loss0.06.hdf5', custom_objects={
    'Adam': lambda **kwargs: hvd.DistributedOptimizer(keras.optimizers.Adam(**kwargs))
})

shape = (128, 128)

x_test_data = np.load(path + '/x_test.npy')
y_test_data = np.load(path + '/y_test.npy')
x_test = utils.get_data(x_test_data, shape)
y_test = utils.get_data(y_test_data, shape)

y_pred = []
for i in x_test:
    i = np.expand_dims(i, -1)
    y_pred.append(model.predict(i, verbose=1))


file_names = []
for i in range(len(y_pred)):
    x_test_affine = x_test_data[i].affine
    x_test_header = x_test_data[i].header
    file_name = os.path.basename(x_test_data[i].file_map['image'].filename)
    file_names.append(file_name)
    # temp = np.moveaxis(y_pred[i], 0, 2)
    img = nib.Nifti1Image(y_pred[i], x_test_affine, x_test_header)
    nib.save(img, os.path.join(save_dir, 'mask_' + file_name))



output = []
for i in range(len(y_test)):
    output.append(np.squeeze(y_pred[i]))


# y_pred = np.load('results/y_pred.npy')

output = []

for i in range(len(y_test)):
    output.append(np.squeeze(y_pred[i]))

utils.save_datavisualisation3(x_test, y_test, output, 'results/', index_first=True, normalized= False)

