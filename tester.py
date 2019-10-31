import numpy as np
from tensorflow import keras
import utils
import os
import data_loader as dl

path = '/Users/Hendrik/Desktop/mlebe/'
path = 'results/'

model = keras.models.load_model(path + 'unet_ep01_val_loss5.46.hdf5')

x_test = np.load(path + '/x_test.npy')
y_test = np.load(path + '/y_test.npy')

y_pred = []
for i in x_test:
    i = np.expand_dims(i, -1)
    y_pred.append(model.predict(i, verbose=1))
output = []
for i in range(len(y_test)):
    output.append(np.squeeze(y_pred[i]))


# y_pred = np.load('results/y_pred.npy')

output = []

for i in range(len(y_test)):
    output.append(np.squeeze(y_pred[i]))

utils.save_datavisualisation3(x_test, y_test, output, 'results/', index_first=True, normalized= False)

