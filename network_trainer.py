import data_loader as dl
import utils
# import bids_tester
import unet
import pickle
import tensorflow as tf
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
import copy
import warnings

def training(data_gen_args, epochs, loss, shape, x_train, y_train, x_val, y_val, x_test, y_test, save_dir, x_test_data, model, seed, Adam, callbacks, slice_view, augmentation = True, visualisation = False, last_step = False):
    """
    Trains the model

    :param data_gen_args:
    :param epochs:
    :param loss:
    :param remote:
    :param shape:
    :param x_train:
    :param y_train:
    :param x_val:
    :param y_val:
    :param x_test:
    :param y_test:
    :param save_dir:
    :param x_test_data:
    :param min_epochs:
    :param model:
    :param seed:
    :param Adam:
    :param reduce_lr:
    :param model_checkpoint:
    :param bidstest_callback:
    :param earlystopper:
    :return: Bool: (False if early stopped before min_epochs, False else), history
    """

    class myModelCheckpoint(keras.callbacks.Callback):
        """Save the model after every epoch.
        `filepath` can contain named formatting options,
        which will be filled with the values of `epoch` and
        keys in `logs` (passed in `on_epoch_end`).
        For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
        then the model checkpoints will be saved with the epoch number and
        the validation loss in the filename.
        # Arguments
            filepath: string, path to save the model file.
            monitor: quantity to monitor.
            verbose: verbosity mode, 0 or 1.
            save_best_only: if `save_best_only=True`,
                the latest best model according to
                the quantity monitored will not be overwritten.
            save_weights_only: if True, then only the model's weights will be
                saved (`model.save_weights(filepath)`), else the full model
                is saved (`model.save(filepath)`).
            mode: one of {auto, min, max}.
                If `save_best_only=True`, the decision
                to overwrite the current save file is made
                based on either the maximization or the
                minimization of the monitored quantity. For `val_acc`,
                this should be `max`, for `val_loss` this should
                be `min`, etc. In `auto` mode, the direction is
                automatically inferred from the name of the monitored quantity.
            period: Interval (number of epochs) between checkpoints.
        """

        def __init__(self, filepath, monitor='val_loss', verbose=0,
                     save_best_only=False, save_weights_only=False,
                     mode='auto', period=20):
            super(myModelCheckpoint, self).__init__()
            self.monitor = monitor
            self.verbose = verbose
            self.filepath = filepath
            self.save_best_only = save_best_only
            self.save_weights_only = save_weights_only
            self.period = period
            self.epochs_since_last_save = 0

            if mode not in ['auto', 'min', 'max']:
                warnings.warn('ModelCheckpoint mode %s is unknown, '
                              'fallback to auto mode.' % (mode),
                              RuntimeWarning)
                mode = 'auto'

            if mode == 'min':
                self.monitor_op = np.less
                self.best = np.Inf
            elif mode == 'max':
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                    self.monitor_op = np.greater
                    self.best = -np.Inf
                else:
                    self.monitor_op = np.less
                    self.best = np.Inf

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            self.epochs_since_last_save += 1
            if self.epochs_since_last_save >= self.period:
                self.epochs_since_last_save = 0
                filepath = self.filepath.format(epoch=epoch + 1, **logs)
                if self.save_best_only:
                    current = logs.get(self.monitor)
                    if current is None:
                        warnings.warn('Can save best model only with %s available, '
                                      'skipping.' % (self.monitor), RuntimeWarning)
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                      ' saving model to %s'
                                      % (epoch + 1, self.monitor, self.best,
                                         current, filepath))
                            self.best = current
                            if self.save_weights_only:
                                self.model.save_weights(filepath, overwrite=True)
                            else:
                                self.model.save(filepath, overwrite=True)
                        else:
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s did not improve from %0.5f' %
                                      (epoch + 1, self.monitor, self.best))
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                    if self.save_weights_only:
                        self.model.save_weights(filepath, overwrite=True)
                    else:
                        self.model.save(filepath, overwrite=True)


    model_checkpoint = myModelCheckpoint(save_dir + 'unet_ep{epoch:02d}_val_loss{val_loss:.2f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, period=20)

    if last_step == True:
        callbacks.append(model_checkpoint)
    try :
        import subprocess
        git_commit_version = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    except Exception as e:
        print(e)


    experiment_description = open(save_dir + 'experiment_description.txt', 'w+')
    experiment_description.write('Git commit Version: {} \n\n'.format(git_commit_version))
    experiment_description.write(
        "This experiment was run on {date_time} \n\n".format(date_time=datetime.datetime.now()))
    if augmentation == True:
        experiment_description.write('Augmentation values: ' + str(data_gen_args.items()) + '\n\n')

    experiment_description.write('Callbacks: {callback}'.format(callback=str(callbacks)) + '\n\n')
    experiment_description.write('Seed: {seed}'.format(seed=seed) + '\n\n')
    experiment_description.write('Shape: {shape}'.format(shape=shape) + '\n\n')
    experiment_description.write('Epochs: {epochs}'.format(epochs=epochs) + '\n\n')
    experiment_description.close()

    model.compile(loss=loss, optimizer=Adam, metrics=['accuracy'])

    if augmentation == True:

        """
        Data augmentation
        """

        augment_save_dir = save_dir + 'augment'
        if not os.path.exists(augment_save_dir):
            os.makedirs(augment_save_dir)


        image_datagen = kp.image.ImageDataGenerator(**data_gen_args)
        mask_datagen = kp.image.ImageDataGenerator(**data_gen_args)
        image_val_datagen = kp.image.ImageDataGenerator(**data_gen_args)
        mask_val_datagen = kp.image.ImageDataGenerator(**data_gen_args)

        image_generator = image_datagen.flow(x_train, seed = seed)
        mask_generator = mask_datagen.flow(y_train, seed = seed)
        image_val_generator = image_val_datagen.flow(x_val, seed = seed+1)
        mask_val_generator = mask_val_datagen.flow(y_val, seed = seed+1)


        imgs = [next(image_generator) for _ in range(1000)]   # number of augmented images
        masks = [next(mask_generator) for _ in range(1000)]
        imgs_val = [next(image_val_generator) for _ in range(1000)]
        masks_val = [next(mask_val_generator) for _ in range(1000)]

        imgs = np.concatenate(imgs)
        masks = np.concatenate(masks)
        imgs_val = np.concatenate(imgs_val)
        masks_val = np.concatenate(masks_val)
        np.save(save_dir + 'x_train_augmented', imgs[:50])
        np.save(save_dir + 'y_train_augmented', masks[:50])
        for i in range(30):
            plt.imshow(np.squeeze(imgs[i,...]), cmap = 'gray')
            plt.imshow(np.squeeze(masks[i,...]), alpha = 0.6, cmap = 'Blues')
            plt.axis('off')
            plt.savefig(augment_save_dir+'/img_{}.pdf'.format(i), format = 'pdf')
            plt.close()
            plt.imshow(np.squeeze(imgs_val[i,...]), cmap = 'gray')
            plt.imshow(np.squeeze(masks_val[i,...]), alpha = 0.6, cmap = 'Blues')
            plt.axis('off')
            plt.savefig(augment_save_dir+'/val_{}.pdf'.format(i), format = 'pdf')
            plt.close()

        train_dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(imgs), tf.data.Dataset.from_tensor_slices(masks)))
        train_dataset = train_dataset.repeat().shuffle(1000).batch(32)
        validation_set = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(imgs_val), tf.data.Dataset.from_tensor_slices(masks_val)))
        validation_set = validation_set.repeat().shuffle(1000).batch(32)

        history = model.fit(train_dataset, steps_per_epoch= int(len(x_train) / 32), validation_data = validation_set, epochs=epochs, validation_steps = int(len(x_train) / 32), verbose=1, callbacks=callbacks)

    else:
        if visualisation:
            if not os.path.exists(save_dir + 'train/'):
                os.makedirs(save_dir + 'train/')
            if not os.path.exists(save_dir + 'val/'):
                os.makedirs(save_dir + 'val/')

            for i in range(x_train.shape[0]):
                plt.imshow(np.squeeze(x_train[i, ...]), cmap='gray')
                plt.imshow(np.squeeze(y_train[i, ...]), alpha=0.6, cmap='Blues')
                plt.savefig(save_dir + 'train/img_{}.pdf'.format(i), format= 'pdf')
                plt.close()
            for i in range(x_val.shape[0]):
                plt.imshow(np.squeeze(x_val[i, ...]), cmap='gray')
                plt.imshow(np.squeeze(y_val[i, ...]), alpha=0.6, cmap='Blues')
                plt.savefig(save_dir + 'val/val_{}.pdf'.format(i), format = 'pdf')
                plt.close()
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                            epochs=epochs, validation_steps=int(len(x_train) / 32), verbose=1,
                            callbacks=callbacks)

    print(history.history.keys())

    plt.figure()

    # Plot training & validation accuracy values:
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
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

    model.save(save_dir + 'model_ep{}.h5'.format(len(history.epoch)), overwrite=True)

    y_pred = []
    dice_scores_string = []
    dice_scores = []
    for x, y in zip(x_test, y_test):
        img_pred = np.empty((x.shape))
        dice_score_img = []
        for slice in range(x.shape[0]):
            temp = np.expand_dims(x[slice],-1) #expand dims for channel
            temp = np.expand_dims(temp, 0) #expand dims for batch
            prediction = model.predict(temp, verbose=0)
            prediction = np.squeeze(prediction)
            prediction_bin = np.where(prediction > 0.9, 1, 0)
            img_pred[slice, ...] = prediction_bin
            dice_scores.append(su.dice(y[slice], prediction_bin))
            dice_score_img.append('dice: ' + str(np.round(su.dice(y[slice], np.where(prediction > 0.6, 1, 0)), 4)))
        y_pred.append(img_pred)
        dice_scores_string.append(dice_score_img)

    temp = np.concatenate(y_pred, 0)
    dice_score = np.mean(dice_scores)
    plt.figure()
    plt.hist(np.unique(temp))
    plt.title('Histogram of the pixel values from the predicted masks')
    plt.savefig(os.path.join(save_dir, 'hist'+str(np.round(dice_score, 4)) + '.png'))
    plt.close()


    print('mean Dice score: ', dice_score)
    file_names = []
    for i in range(len(y_pred)):
        x_test_affine = x_test_data[i].affine
        x_test_header = x_test_data[i].header
        file_name = os.path.basename(x_test_data[i].file_map['image'].filename)
        file_names.append(file_name)

        y_true_temp = np.moveaxis(y_test[i], 0, 2)
        y_true_temp = nib.Nifti1Image(y_true_temp, x_test_affine, x_test_header)
        nib.save(y_true_temp, os.path.join(save_dir, 'y_true_' + file_name))
        mask_temp = np.moveaxis(y_pred[i], 0, 2)
        img_temp = np.moveaxis(x_test[i], 0, 2)
        img = nib.Nifti1Image(img_temp, x_test_affine, x_test_header)
        nib.save(img, os.path.join(save_dir, 'img_' + file_name))
        mask = nib.Nifti1Image(mask_temp, x_test_affine, x_test_header)
        nib.save(mask, os.path.join(save_dir, 'mask_' + file_name))

    thresholds = [0, 0.9]
    outputs = []
    row_titles = ['x_test', 'y_test', 'raw prediction', 'thr 0.5', 'thr 0.7', 'thr 0.8', 'thr 0.9']
    slice_titles = [None, None, dice_scores_string, None, None, None, None]
    for thr in thresholds:
        if thr == 0:
            outputs.append([np.squeeze(img) for img in y_pred])
        else:
            outputs.append([np.where(np.squeeze(img) > thr, 1, 0) for img in y_pred])
    list = [x_test, y_test]
    for o in outputs:
        list.append(o)
    utils.save_datavisualisation_plt(list, save_dir, normalized=True, file_names=file_names, figure_title = 'Predictions with a median Dice score of {}'.format(np.round(dice_score,4)), slice_titles=slice_titles)

    np.save(save_dir + 'y_pred_{}dice'.format(np.round(dice_score, 4)), y_pred)

    return history

def network_trainer(file_name, test, loss, epochss, shape, data_gen_argss, blacklist, data_type, slice_view,visualisation = False, pretrained_model = False):
    """
    This function loads the data, preprocesses it and trains the network with given parameters.
    It trains the network successively with different data augmentation values.
    If the training is early stopped before 'min_epochs', the training is started again with reduced augmetnation values

    :param test: Bool: If Test is True, every parameter is set to increase learning speed. Used to test if the code runs
    :param remote: Bool: If remote is True, the paths are set for remote computer
    :param loss: string: loss with which the model will be trained
    :param epochss: Array with epochs. Should have the same length than data_gen_argss
    :param shape: Tuple (y,x): Shape of the images that should come out of the preprocessing
    :param data_gen_argss: Array of dicts : arguments for the data augmentations, should have the same length than epochss
    :param min_epochs: int: The minimum amount of epochs the network should be trained on. If this number is not reached, the training will start again with reduced augmentation values
    :param max_tries: int: Integer indicating how many times the training should be started again with reduced augmentation values
    :param visualisation: Bool: if True, all images after preprocessing are saved
    :return: Bool: True if min_epochs is not reached, False otherwise
    """

    seed = random.randint(0, 1000)

    print('Training with seed: ', seed)

    image_dir_remote = '/mnt/data/mlebe_data/'
    data_dir = '/usr/share/mouse-brain-atlases/'
    if data_type == 'anat':
        img_data = dl.load_img_remote(image_dir_remote, blacklist)
    elif data_type == 'func':
        img_data = dl.load_func_img(image_dir_remote, blacklist)


    if test == True:
        epochss = np.ones(len(data_gen_argss), dtype=int)
        save_dir = 'test/{loss}_{epochs}_{date}/'.format(
            loss=loss, epochs=np.sum(epochss), date=datetime.date.today())
        import shutil
        if os.path.exists('test/'):
            shutil.rmtree('test/')


    else:
        save_dir = image_dir_remote + 'results/' + file_name + '/{loss}_{epochs}_{date}/'.format(loss=loss,epochs=np.sum(epochss),date=datetime.date.today())


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print('creating dir: ', save_dir)


    """shape = (z,y,x)"""

    temp = dl.load_mask(data_dir)
    mask_data = []


    for i in range(len(img_data)):
        mask_data.append(copy.deepcopy(temp[0]))

    # utils.get_image_and_mask(img_data,mask_data, shape,  save_dir, remove_black_labels_and_columns, slice_view)           #with this line all the images with the mask can be saved to create a blacklist


    print('*** Splitting data into Train, Validation and Test set ***')
    if test == True:
        x_train1_data, x_test_data, y_train1_data, y_test_data = model_selection.train_test_split(img_data, mask_data,
                                                                                                  random_state=seed,
                                                                                                  test_size=0.9, shuffle = True)
    else:
        x_train1_data, x_test_data, y_train1_data, y_test_data = model_selection.train_test_split(img_data, mask_data,
                                                                                                  random_state=seed,
                                                                                                  test_size=0.1, shuffle = True)

    print('*** Preprocessing ***')

    x_train1, y_train1, x_train1_affines, x_train1_headers, x_train1_file_names, = utils.get_image_and_mask(x_train1_data, y_train1_data, shape, save_dir, slice_view= slice_view, visualisation=visualisation, blacklist_bool = blacklist)[:5]
    x_test, y_test, x_test_affines, x_test_headers, file_names, y_test_affines, y_test_headers = utils.get_image_and_mask(x_test_data, y_test_data, shape, save_dir,slice_view= slice_view, visualisation=visualisation, blacklist_bool = blacklist)


    x_train_struct = {
        'x_train': x_train1,
        'x_train_affines': x_train1_affines,
        'x_train_headers': x_train1_headers,
        'file_names': x_train1_file_names,
    }

    y_train_struct = {
        'y_train': y_train1
    }

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
    xfile = open(save_dir + 'y_train_struct.pkl', 'wb')
    pickle.dump(y_train_struct, xfile)
    xfile.close()
    xfile = open(save_dir + 'x_train_struct.pkl', 'wb')
    pickle.dump(x_train_struct, xfile)
    xfile.close()

    x_train1 = np.concatenate(x_train1, axis=0)
    y_train1 = np.concatenate(y_train1, axis=0)
    x_train1 = np.expand_dims(x_train1, -1)
    y_train1 = np.expand_dims(y_train1, -1)
    x_train, x_val, y_train, y_val = model_selection.train_test_split(x_train1, y_train1, test_size=0.25, shuffle= True, random_state= seed)


    print('TRAINING SHAPE: ' + str(x_train.shape[1:4]))
    print('*** Training with {} slices ***'.format(x_train.shape[0]))
    print('*** Validating with {} slices ***'.format(x_val.shape[0]))
    input_shape = (x_train.shape[1:4])

    """
    Callbacks
    """


    class bidstest(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, log={}):
            if epoch % 10 == 0:
                if bids_tester.bids_tester(new_save_dir, self.model, remote, shape, slice_view = slice_view, epochs= epoch):  # Test should be True (default) to only predict 5 bids_images instead of all of them
                    print('No faulty predictions!')
                    return
                else:  # if predictions are zero, training stops
                    self.model.stop_training = True

    # bidstest_callback = bidstest()

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, verbose=1, patience=5)
    earlystopper = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
    Adam = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, amsgrad=True)

    callbacks = [reduce_lr, earlystopper]
    # callbacks = [bidstest_callback, reduce_lr, earlystopper]          #todo comment this to remove cv2 dependency

    if loss == 'bincross':
        loss = 'binary_crossentropy'

    elif loss == 'dice':
        print('\n*********\n\nTraining with loss: dice-loss\n\n*********\n')
        loss = unet.dice_coef_loss

    elif loss == 'dice_bincross':
        print('\n*********\n\nTraining with loss: dice_bincross\n\n*********\n')
        loss = unet.dice_bincross_loss

    elif loss == 'thr_bincross':
        print('\n*********\n\nTraining with loss: thr_bincross\n\n*********\n')
        loss = unet.thr_dice_coef

    else:
        print('\n*********\n\nWrong loss function, choose between bincross, dice or dice_bincross\n\n*********\n')

    if pretrained_model == False:
        if test == True:
            model = unet.twolayernetwork(input_shape, 3, 0.5)

        else:
            model = unet.unet(input_shape)
    else: model = keras.models.load_model(pretrained_model, custom_objects = {'dice_coef_loss': unet.dice_coef_loss})

    """
    Training
    Two loop variables: 
    - counter counts the number of steps
    - nmbr_tries counts the number of tries per step
    """
    counter = 1

    for data_gen_args, epochs in zip(data_gen_argss, epochss):

        if data_gen_args == None:
            augmentation = False
        else: augmentation = True


        if counter > 1:
            print('\n\n\n\n********* \nTraining with higher augmentation values! Taking model from try {} \n*********\n\n\n\n'.format(best_try + 1))
            model = history.model

        histories = []
        new_save_dir = save_dir + '{counter}_Step/'.format( counter = counter)

        if not os.path.exists(new_save_dir):
            os.makedirs(new_save_dir)
        if data_gen_args == data_gen_argss[-1]:
            last_step = True
            # callbacks[-1] = EarlyStopping(monitor='val_loss', patience=80, verbose=1)
        else: last_step = False

        print('Step', counter, 'of', len(epochss))
        temp_history = training(data_gen_args, epochs, loss, shape, x_train, y_train, x_val, y_val, x_test, y_test, new_save_dir, x_test_data, model, seed, Adam, callbacks, slice_view, augmentation= augmentation, visualisation=visualisation, last_step = last_step)
        histories.append(temp_history)

        history_epochs = []
        for x in histories:
            history_epochs.append(len(x.epoch))
            best_try = history_epochs.index(max(history_epochs))    #best_try is the try with the most epochs
        history = histories[best_try]
        counter += 1





