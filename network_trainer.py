import data_loader as dl
import utils
import bids_tester


import pickle
import unet
import scoring_utils as su
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import *
import tensorflow.keras.preprocessing as kp
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn import model_selection
import nibabel as nib
import datetime
import random

#todo write txt file with comment to experiment
#todo verify augmentation values
#todo parse arguments?
#todo calculate dice score
#todo write README
#todo write scratches with useful functions


test = False
remote = True
visualisation = False  #if visualisation true saves pre- and unpreprocessed images for visualisation
# losses = ['dice_bincross', 'dice', 'bincross']
losses = ['dice_bincross']
epochs = 300
seed = random.randint(0, 1000)
shape = (128, 128)



for loss in losses:


    if remote == True:
            image_dir_remote = '/mnt/scratch/'
            data_dir = '/usr/share/mouse-brain-atlases/'
            img_data = dl.load_img_remote(image_dir_remote)

    else:
        image_dir = '/Users/Hendrik/Documents/mlebe_data/preprocessed'
        img_data = dl.load_img(image_dir)
        data_dir = '/Users/Hendrik/Documents/mlebe_data/mouse-brain-atlases/'  # local

    if test == True:
        epochs = 1
        save_dir = '/Users/Hendrik/Documents/mlebe_data/results/test/'
    else:
        save_dir = 'results/training_results/{loss}_{epochs}_{date}/'.format(loss = loss, epochs = epochs, date = datetime.date.today())

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_gen_args = dict(rotation_range=90,
                        zca_whitening = True,
                        zca_epsilon = 1e-6,
                        brightness_range = [0.5, 1.2],
                        width_shift_range=30,
                        height_shift_range=30,
                        shear_range= 5,
                        zoom_range= 0.2,
                        horizontal_flip=True,
                        vertical_flip = True,
                        fill_mode='nearest')

    experiment_description = open(save_dir + 'experiment_description.txt', 'w+')
    experiment_description.write("This experiment was run on {date_time} \n\n".format(date_time = datetime.datetime.now()))
    experiment_description.write('Augmentation values: ' + str(data_gen_args.items()) + '\n\n')
    experiment_description.write('Seed: {seed}'.format(seed = seed) + '\n\n')
    experiment_description.close()
    """shape = (z,y,x)"""


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
    x_test, x_test_affines, x_test_headers, file_names = utils.get_data(x_test_data, shape, save_dir)
    y_test, y_test_affines, y_test_headers = utils.get_data(y_test_data, shape, save_dir)[:3]

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
    model_checkpoint = ModelCheckpoint(save_dir + '/unet_ep{epoch:02d}_val_loss{val_loss:.2f}.hdf5', monitor='loss', verbose = 1, save_best_only=True, period = 10)

    class bidstest(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, log = {}):
            if epoch % 10 == 0:
                bids_tester.bids_tester(save_dir, self.model, remote, shape, epoch) #Test should be True (default) to only predict 5 bids_images instead of all of them
            return

    bidstest_callback = bidstest()

    if test == True:
        model = unet.twolayernetwork(input_shape, 3, 0.5)
    else:
        model = unet.unet(input_shape)

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
    elif loss == 'dice_bincross':
        print('Training with loss: dice_bincross')
        model.compile(loss=unet.dice_bincross_loss,
                      optimizer='adam',
                      metrics=['accuracy'])
    else: print('wrong loss function, choose between bincross, dice or dice_bincross')


    aug = kp.image.ImageDataGenerator(**data_gen_args)


    history = model.fit_generator(aug.flow(x_train, y_train), steps_per_epoch=len(x_train) / 32, validation_data=(x_val, y_val), epochs=epochs, verbose=1, callbacks=[model_checkpoint, bidstest_callback])



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

    bids_tester.bids_tester(save_dir, model, remote, shape, epochs)

    utils.save_datavisualisation3(x_test, y_test, output, save_dir , index_first = True, normalized = True, file_names = file_names)


    np.save(save_dir + 'y_pred_{}dice'.format(np.round(dice_score, 4)), y_pred)
