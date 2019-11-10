import data_loader as dl
import pickle
import unet
import utils
import scoring_utils as su
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import *
import tensorflow.keras.preprocessing as kp
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn import model_selection
import nibabel as nib

#todo calculate dice score
#todo write README
#todo write scratches with useful functions


test = False
remote = False
visualisation = True  #if visualisation true saves pre- and unpreprocessed images for visualisation
loss = 'bincross'
epochs = 50
seed = 1
shape = (128, 128)

if test == True:
    epochs = 1
    save_dir = '/Users/Hendrik/Documents/mlebe_data/results/test/'
else:
    save_dir = 'results/training_results/{loss}_{epochs}/'.format(loss = loss, epochs = epochs)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

"""shape = (z,y,x)"""

if remote == True:
    img_data = dl.load_img_remote()
    data_dir = '/usr/share/mouse-brain-atlases/'
else:
    img_data = dl.load_img()
    data_dir = '/Users/Hendrik/Documents/mlebe_data/mouse-brain-atlases/'  # local


temp = dl.load_mask(data_dir)
mask_data = []

for i in range(len(img_data)):
    mask_data.append(temp[0])

print('*** splitting data into Train, Validation and Test set ***')
if test == True:
    x_train1_data, x_test_data , y_train1_data, y_test_data = model_selection.train_test_split(img_data, mask_data, random_state = seed, test_size=0.9)
else:
    x_train1_data, x_test_data, y_train1_data, y_test_data = model_selection.train_test_split(img_data, mask_data, random_state = seed, test_size=0.1)


print('*** Preprocessing ***')
x_train1 = utils.get_data(x_train1_data,shape, save_dir, visualisation = visualisation)[0]
y_train1 = utils.get_data(y_train1_data,shape, save_dir, visualisation = visualisation)[0]
x_test, x_test_affines, x_test_headers, file_names = utils.get_data(x_test_data, shape)
y_test, y_test_affines, y_test_headers = utils.get_data(y_test_data, shape)[:3]  #todo ca fait aucun sens de preprocess 10 fois les memes masks

print('*** saving Test data ***')
x_test_struct = {
    'x_test' : x_test,
    'x_test_affines' : x_test_affines,
    'x_test_headers' : x_test_headers,
    'file_names' : file_names
}

y_test_struct = {
    'y_test' : y_test,
    'y_test_affines' : y_test_affines,
    'y_test_headers' : y_test_headers,
}

xfile = open(save_dir + 'x_test_struct.pkl', 'wb')
pickle.dump(x_test_struct, xfile)
xfile.close()
yfile = open(save_dir + 'y_test_struct.pkl', 'wb')
pickle.dump(y_test_struct, yfile)
yfile.close()

x_train1 = np.concatenate(x_train1, axis = 0)
y_train1 = np.concatenate(y_train1, axis = 0)
x_train1 = np.expand_dims(x_train1, -1)
y_train1 = np.expand_dims(y_train1, -1)
x_train, x_val, y_train, y_val = model_selection.train_test_split(x_train1, y_train1, test_size=0.25)


print('TRAINING SHAPE: ' + str(x_train.shape[1:4]))
input_shape = (x_train.shape[1:4])
model_checkpoint = ModelCheckpoint(save_dir + 'unet_ep{epoch:02d}_val_loss{val_loss:.2f}.hdf5', monitor='loss', verbose=1, save_best_only=True)
if test == True:
    model = unet.twolayernetwork(input_shape, 3, 0.5)
    if loss == 'bincross':
        print('Training with loss: binary_crossentropy')
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    elif loss == 'dice':
        print('Training with loss: dice-loss')
        model.compile(loss = unet.dice_coef_loss,
                      optimizer='adam',
                      metrics=['accuracy'])
    else: print('wrong loss function, choose between bincross or dice')

else:
    model = unet.unet(input_shape)
    if loss == 'bin_cross':
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    if loss == 'Dice':
        model.compile(optimizer=Adam(lr=1e-4), loss=unet.dice_coef_loss, metrics=['accuracy'])



aug = kp.image.ImageDataGenerator(**data_gen_args)


history = model.fit_generator(aug.flow(x_train, y_train), steps_per_epoch=len(x_train) / 32, validation_data=(x_val, y_val), epochs=epochs, verbose=1, callbacks=[model_checkpoint])



print(history.history.keys())
plt.figure()

# Plot training & validation accuracy values:
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
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
plt.savefig(os.path.join(save_dir, 'loss_values.png'))
plt.close()



y_pred = []
dice_scores = []
for i in x_test:
    i = np.expand_dims(i, -1)
    temp = model.predict(i, verbose=1)
    y_pred.append(temp)
    dice_scores.append(su.dice(i, temp))

dice_score = np.median(dice_scores)
print('median Dice score: ', dice_score)
file_names = []
for i in range(len(y_pred)):
    x_test_affine = x_test_data[i].affine
    x_test_header = x_test_data[i].header
    file_name = os.path.basename(x_test_data[i].file_map['image'].filename)
    file_names.append(file_name)
    temp = np.moveaxis(y_pred[i], 0, 2)
    img = nib.Nifti1Image(temp, x_test_affine, x_test_header)
    nib.save(img, os.path.join(save_dir, 'mask_' + file_name))

output = []
for i in range(len(y_test)):
    output.append(np.squeeze(y_pred[i]))

utils.save_datavisualisation3(x_test, y_test, output, save_dir , index_first = True, normalized = True, file_names = file_names)


np.save(save_dir + 'y_pred_{}dice'.format(np.round(dice_score, 4)), y_pred)

