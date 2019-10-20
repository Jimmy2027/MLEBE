import data_loader as dl
import model
import utils
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from model import *
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn import model_selection



test = True
remote = False
visualisation = False
epochs = 1

"""shape = (z,x,y)"""

if remote == True:
    img_data = dl.load_img_remote()
    data_dir = '/usr/share/mouse-brain-atlases/'
else:
    import torch
    img_data = dl.load_img(visualisation)
    data_dir = '/Users/Hendrik/Documents/mlebe_data/mouse-brain-atlases/'  # local

temp = dl.load_mask(data_dir, visualisation)
mask_data = []
for i in range(len(img_data)):
    mask_data.append(temp[0])

x_train, x_test , y_train, y_test = model_selection.train_test_split(img_data, mask_data, test_size=0.3)

# torch.save(x_test, 'x_test')
# torch.save(y_test, 'y_test')


x_train = np.concatenate(x_train, axis = 0)
y_train = np.concatenate(y_train, axis = 0)
x_train = np.expand_dims(x_train, -1)
y_train = np.expand_dims(y_train, -1)

input_shape = (x_train.shape[1:4])
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
if test == True:
    model = model.twolayernetwork(input_shape, 3, 0.5)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

else:
    model = model.unet(input_shape)

history = model.fit(x_train, y_train, epochs=epochs, validation_split=0.25, callbacks=[model_checkpoint])

save_dir = 'results/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

print(history.history.keys())
plt.figure()
# Plot training & validation accuracy values:
if remote == True:
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
else:
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(os.path.join(save_dir, 'accuracy_values.png'))
plt.close()

plt.figure()
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(os.path.join(save_dir, 'epochs_loss_values.png'))
plt.close()


model.save(save_dir + 'test.h5')

y_pred = []
for i in x_test:
    i = np.expand_dims(i, -1)
    y_pred.append(model.predict(i, verbose=1))
output = []
for i in range(len(y_test)):
    output.append(np.squeeze(y_pred[i]))

utils.save_datavisualisation3(x_test, y_test, output, 'results/', index_first=True, normalized= True)


np.save(save_dir + 'y_pred', y_pred)

