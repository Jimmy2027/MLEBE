import data_loader as dl
import model
import os
import numpy as np
from matplotlib import pyplot as plt

mask_data = dl.load_mask()

img_data = dl.load_img()

x_train = np.concatenate((img_data[0], img_data[1]), 2)
x_train = np.reshape(x_train, (x_train.shape[2], x_train.shape[0], x_train.shape[1], x_train.shape[3]))
y_train = np.concatenate((mask_data[0], mask_data[1]), 2)
y_train = np.reshape(y_train, (y_train.shape[2], y_train.shape[0], y_train.shape[1], y_train.shape[3]))
x_test = img_data[2]
x_test = np.reshape(x_test, (x_test.shape[2], x_test.shape[0], x_test.shape[1], x_test.shape[3]))
y_test = mask_data[2]
y_test = np.reshape(y_test, (y_test.shape[2], y_test.shape[0], y_test.shape[1],y_test.shape[3]))

model = model.unet()

history = model.fit(x_train, y_train, epochs = 5)

print(history.history.keys())

model.save('test.h5')

y_pred = []
for i in x_test:
    y_pred.append(model.predict(i, verbose=1))

np.save(os.path.join('y_pred'), y_pred)

