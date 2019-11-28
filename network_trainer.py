import data_loader as dl
import utils
import bids_tester
import unet


import pickle
import tensorflow
import scoring_utils as su
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import *
import tensorflow.keras.preprocessing as kp
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn import model_selection
import nibabel as nib
import datetime
import random

#todo verify augmentation values
#todo parse arguments?
#todo write README
#todo write scratches with useful functions


def network_trainer(test, remote, loss, epochss, shape, nmbr_tries, data_gen_argss, min_epochs, visualisation = False):
    """
    This function loads the data, preprocesses it and trains the network with given parameters.
    It trains the network successively with different data augmentation values.

    :param test: Bool: If Test is True, every parameter is set to increase learning speed. Used to test is the code runs
    :param remote: Bool: If remote is True, the paths are set for remote computer
    :param loss: Array of strings: with which loss the network will be trained
    :param epochss: Array with epochs. Should have the same length than data_gen_argss
    :param shape: Tuple (y,x): Shape of the images that should come out of the preprocessing
    :param data_gen_argss: Array of dicts : arguments for the data augmentations, should have the same length than epochss
    :param min_epochs: int: The minimum amount of epochs the network should be trained on. If this number is not reached, the training will start again with a different seed
    :param visualisation: Bool: if True, all images after preprocessing are saved
    :return: Bool: True if min_epochs is not reached, False otherwise
    """

    seed = random.randint(0, 1000)
    print('Training with seed: ', seed)
    if remote == True:
        image_dir_remote = '/mnt/scratch/'
        data_dir = '/usr/share/mouse-brain-atlases/'
        img_data = dl.load_img_remote(image_dir_remote)

    else:
        image_dir = '/Users/Hendrik/Documents/mlebe_data/preprocessed'
        img_data = dl.load_img(image_dir)
        data_dir = '/Users/Hendrik/Documents/mlebe_data/mouse-brain-atlases/'  # local

    if test == True:
        epochss = np.ones(len(data_gen_argss), dtype=int)
        min_epochs = 0
        save_dir = '/Users/Hendrik/Documents/mlebe_data/results/test/{loss}_{epochs}_{date}_try{tries}/'.format(
            loss=loss, epochs=np.sum(epochss), date=datetime.date.today(), tries=nmbr_tries)
    else:
        save_dir = 'with_augment_successiv/training_results/{loss}_{epochs}_{date}_try{tries}/'.format(loss=loss,
                                                                                                       epochs=np.sum(
                                                                                                           epochss),
                                                                                                       date=datetime.date.today(),
                                                                                                       tries=nmbr_tries)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    experiment_description = open(save_dir + 'experiment_description.txt', 'w+')
    experiment_description.write(
        "This experiment was run on {date_time} \n\n".format(date_time=datetime.datetime.now()))
    for i, data_gen_args in enumerate(data_gen_argss):
        experiment_description.write('Augmentation_{} values: '.format(i) + str(data_gen_args.items()) + '\n\n')
    experiment_description.write('Seed: {seed}'.format(seed=seed) + '\n\n')
    experiment_description.close()

    """shape = (z,y,x)"""

    temp = dl.load_mask(data_dir)
    mask_data = []

    for i in range(len(img_data)):
        mask_data.append(temp[0])

    print('*** Splitting data into Train, Validation and Test set ***')
    if test == True:
        x_train1_data, x_test_data, y_train1_data, y_test_data = model_selection.train_test_split(img_data, mask_data,
                                                                                                  random_state=seed,
                                                                                                  test_size=0.9)
    else:
        x_train1_data, x_test_data, y_train1_data, y_test_data = model_selection.train_test_split(img_data, mask_data,
                                                                                                  random_state=seed,
                                                                                                  test_size=0.1)

    print('*** Preprocessing ***')
    x_train1 = utils.get_data(x_train1_data, shape, save_dir, visualisation=visualisation)[0]
    y_train1 = utils.get_data(y_train1_data, shape, save_dir, visualisation=visualisation)[0]
    x_test, x_test_affines, x_test_headers, file_names = utils.get_data(x_test_data, shape, save_dir)
    y_test, y_test_affines, y_test_headers = utils.get_data(y_test_data, shape, save_dir)[:3]

    print('*** Saving Test data ***')
    x_test_struct = {
        'x_test': x_test,
        'x_test_affines': x_test_affines,
        'x_test_headers': x_test_headers,
        'file_names': file_names
    }

    y_test_struct = {
        'y_test': y_test,
        'y_test_affines': y_test_affines,
        'y_test_headers': y_test_headers,
    }

    xfile = open(save_dir + 'x_test_struct.pkl', 'wb')
    pickle.dump(x_test_struct, xfile)
    xfile.close()
    yfile = open(save_dir + 'y_test_struct.pkl', 'wb')
    pickle.dump(y_test_struct, yfile)
    yfile.close()

    x_train1 = np.concatenate(x_train1, axis=0)
    y_train1 = np.concatenate(y_train1, axis=0)
    x_train1 = np.expand_dims(x_train1, -1)
    y_train1 = np.expand_dims(y_train1, -1)
    x_train, x_val, y_train, y_val = model_selection.train_test_split(x_train1, y_train1, test_size=0.25)

    print('TRAINING SHAPE: ' + str(x_train.shape[1:4]))
    input_shape = (x_train.shape[1:4])

    class bidstest(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, log={}):
            if epoch % 10 == 0:
                if bids_tester.bids_tester(save_dir, self.model, remote, shape,
                                           epoch):  # Test should be True (default) to only predict 5 bids_images instead of all of them
                    print('No faulty predictions!')
                    return
                else:  # if predictions are zero, training stops
                    self.model.stop_training = True

    """
    Callbacks
    """
    bidstest_callback = bidstest()
    model_checkpoint = ModelCheckpoint(save_dir + '/unet_ep{epoch:02d}_val_loss{val_loss:.2f}.hdf5', monitor='loss',
                                       verbose=1, save_best_only=True, period=10)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, verbose=1, patience=2)
    earlystopper = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)

    Adam = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, amsgrad=True)


    if test == True:
        model = unet.twolayernetwork(input_shape, 3, 0.5)
    else:
        model = unet.unet(input_shape)

    if loss == 'bincross':
        print('Training with loss: binary_crossentropy')
        loss = 'binary_crossentropy'

    elif loss == 'dice':
        print('Training with loss: dice-loss')
        loss = unet.dice_coef_loss

    elif loss == 'dice_bincross':
        print('Training with loss: dice_bincross')
        loss = unet.dice_bincross_loss

    elif loss == 'thr_bincross':
        print('Training with loss: thr_bincross')
        loss = unet.thr_dice_coef

    else:
        print('Wrong loss function, choose between bincross, dice or dice_bincross')

    """
    Training

    """
    counter = 1
    for data_gen_args, epochs in zip(data_gen_argss, epochss):
        if counter > 1:
            print('\n\n\n\n********* \nTraining with higher augmentation values! \n*********\n\n\n\n')
            model = history.model
            save_dir = save_dir + '/{}_Step/'.format(counter)

        model.compile(loss=loss, optimizer=Adam, metrics=['accuracy'])
        augment_save_dir = save_dir + '/augment/'

        aug = kp.image.ImageDataGenerator(**data_gen_args)
        # image_datagen = kp.image.ImageDataGenerator(**data_gen_args)
        # mask_datagen = kp.image.ImageDataGenerator(**data_gen_args)
        #
        # image_generator = image_datagen.flow(x_train, save_to_dir = augment_save_dir)
        #
        # mask_generator = mask_datagen.flow(y_train, save_to_dir = augment_save_dir)
        #
        # train_generator = zip(image_generator, mask_generator)

        # if not os.path.exists(augment_save_dir):
        #     os.makedirs(augment_save_dir)
        history = model.fit_generator(aug.flow(x_train, y_train), steps_per_epoch=len(x_train) / 32,
                                      validation_data=(x_val, y_val), epochs=epochs, verbose=1,
                                      callbacks=[reduce_lr, model_checkpoint, bidstest_callback, earlystopper])

        # history = model.fit_generator(train_generator, steps_per_epoch=len(x_train) / 32, validation_data=(x_val, y_val), epochs=epochs,verbose=1, callbacks=[model_checkpoint, bidstest_callback])

        print(history.history.keys())
        if len(history.epoch) < min_epochs:
            print('Faulty predictions! Epoch:', len(history.epoch), 'instead of', epochs)
            return True

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
            temp = model.predict(i, verbose=0)
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

        utils.save_datavisualisation3(x_test, y_test, output, save_dir, index_first=True, normalized=True,
                                      file_names=file_names)

        np.save(save_dir + 'y_pred_{}dice'.format(np.round(dice_score, 4)), y_pred)

        counter += 1

    return False
