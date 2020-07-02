import copy
import datetime
import os
import pickle
import random
import uuid
import warnings

import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.preprocessing as kp
from matplotlib import pyplot as plt
from sklearn import model_selection
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

import mlebe.training.two_D.attention_unet as attention_unet
import mlebe.training.two_D.unet as unet
import mlebe.training.two_D.utils.data_loader as dl
import mlebe.training.two_D.utils.general as utils
import mlebe.training.two_D.utils.scoring_utils as su
from mlebe.training.two_D.utils.data_augment import augment
from mlebe.training.two_D.utils.masking_vis import tester


def training(experiment_config, trainer_config, data_dir, studies, data_type, data_gen_args, epochs, loss, shape,
             x_train, y_train, x_val, y_val, x_test, y_test, save_dir, x_test_data, model, seed, Adam, callbacks,
             slice_view, augmentation=True, visualisation=False, last_step=False, test=False):
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

    experiment_description = open(save_dir + 'experiment_description.txt', 'w+')
    try:
        import subprocess
        git_commit_version = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
        experiment_description.write('Git commit Version: {} \n\n'.format(git_commit_version))
    except Exception as e:
        print(e)
    experiment_description.write(
        "This experiment was run on {date_time} \n\n".format(date_time=datetime.datetime.now()))
    if augmentation == True:
        experiment_description.write('Augmentation values: ' + str(data_gen_args.items()) + '\n\n')

    experiment_description.write('Callbacks: {callback}'.format(callback=str(callbacks)) + '\n\n')
    experiment_description.write('Seed: {seed}'.format(seed=seed) + '\n\n')
    experiment_description.write('Shape: {shape}'.format(shape=shape) + '\n\n')
    experiment_description.write('Epochs: {epochs}'.format(epochs=epochs) + '\n\n')
    experiment_description.close()
    experiment_config['callbacks'] = str(callbacks)

    if trainer_config.tensorboard_bool:
        logdir = save_dir + 'logs/'
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        tb = keras.callbacks.TensorBoard(log_dir=logdir)
        callbacks.append(tb)

    model.compile(loss=loss, optimizer=Adam, metrics=['accuracy', unet.dice_coef])

    if augmentation == True:

        """
        Data augmentation
        """

        augment_save_dir = save_dir + 'augment'
        if not os.path.exists(augment_save_dir):
            os.makedirs(augment_save_dir)

        data_gen_args_ = {key: data_gen_args[key] for key in data_gen_args.keys() if
                          not key in ['brightness_range', 'noise_var_range', 'bias_var_range']}

        image_datagen = kp.image.ImageDataGenerator(**data_gen_args_)
        mask_datagen = kp.image.ImageDataGenerator(**data_gen_args_)
        image_val_datagen = kp.image.ImageDataGenerator(**data_gen_args_)
        mask_val_datagen = kp.image.ImageDataGenerator(**data_gen_args_)

        image_generator = image_datagen.flow(x_train, seed=seed)
        mask_generator = mask_datagen.flow(y_train, seed=seed)
        image_val_generator = image_val_datagen.flow(x_val, seed=seed + 1)
        mask_val_generator = mask_val_datagen.flow(y_val, seed=seed + 1)

        imgs = [next(image_generator) for _ in range(1000)]
        masks = [np.where(next(mask_generator) > 0.5, 1, 0).astype('float32') for _ in range(
            1000)]  # because keras datagumentation interpolates the data, a threshold must be taken to make the data binary again
        imgs_val = [next(image_val_generator) for _ in range(1000)]
        masks_val = [np.where(next(mask_val_generator) > 0.5, 1, 0).astype('float32') for _ in range(1000)]

        imgs = np.concatenate(imgs)
        masks = np.concatenate(masks)
        imgs_val = np.concatenate(imgs_val)
        masks_val = np.concatenate(masks_val)

        for i in range(imgs.shape[0]):
            imgs[i] = augment(imgs[i], masks[i], brightness_range=data_gen_args['brightness_range'],
                              noise_var_range=data_gen_args['noise_var_range'],
                              bias_var_range=data_gen_args['bias_var_range'])
        for i in range(imgs_val.shape[0]):
            imgs_val[i] = augment(imgs_val[i], masks_val[i], brightness_range=data_gen_args['brightness_range'],
                                  noise_var_range=data_gen_args['noise_var_range'],
                                  bias_var_range=data_gen_args['bias_var_range'])

        np.save(save_dir + 'x_train_augmented', imgs[:50])
        np.save(save_dir + 'y_train_augmented', masks[:50])

        for i in range(100):
            plt.figure()
            plt.subplot(1, 3, 1)
            plt.imshow(np.squeeze(imgs[i, ...]), cmap='gray')
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(np.squeeze(masks[i, ...]), cmap='gray')
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(np.squeeze(imgs[i, ...]), cmap='gray')
            plt.imshow(np.squeeze(masks[i, ...]), alpha=0.6, cmap='Blues')
            plt.axis('off')
            plt.savefig(augment_save_dir + '/img_{}.pdf'.format(i), format='pdf')
            plt.close()
            plt.imshow(np.squeeze(imgs_val[i, ...]), cmap='gray')
            plt.imshow(np.squeeze(masks_val[i, ...]), alpha=0.6, cmap='Blues')
            plt.axis('off')
            plt.savefig(augment_save_dir + '/val_{}.pdf'.format(i), format='pdf')
            plt.close()
        train_dataset = tf.data.Dataset.zip(
            (tf.data.Dataset.from_tensor_slices(imgs), tf.data.Dataset.from_tensor_slices(masks)))
        train_dataset = train_dataset.repeat().shuffle(1000).batch(trainer_config.batch_size)
        validation_set = tf.data.Dataset.zip(
            (tf.data.Dataset.from_tensor_slices(imgs_val), tf.data.Dataset.from_tensor_slices(masks_val)))
        validation_set = validation_set.repeat().shuffle(1000).batch(trainer_config.batch_size)

        history = model.fit(train_dataset, steps_per_epoch=int(len(x_train) / trainer_config.batch_size),
                            validation_data=validation_set,
                            epochs=epochs, validation_steps=int(len(x_train) / trainer_config.batch_size), verbose=1,
                            callbacks=callbacks)

    else:
        if visualisation:
            if not os.path.exists(save_dir + 'train/'):
                os.makedirs(save_dir + 'train/')
            if not os.path.exists(save_dir + 'val/'):
                os.makedirs(save_dir + 'val/')

            for i in range(x_train.shape[0]):
                plt.imshow(np.squeeze(x_train[i, ...]), cmap='gray')
                plt.imshow(np.squeeze(y_train[i, ...]), alpha=0.6, cmap='Blues')
                plt.savefig(save_dir + 'train/img_{}.pdf'.format(i), format='pdf')
                plt.close()
            for i in range(x_val.shape[0]):
                plt.imshow(np.squeeze(x_val[i, ...]), cmap='gray')
                plt.imshow(np.squeeze(y_val[i, ...]), alpha=0.6, cmap='Blues')
                plt.savefig(save_dir + 'val/val_{}.pdf'.format(i), format='pdf')
                plt.close()

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(len(x_train)).batch(batch_size=trainer_config.batch_size).repeat()
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = val_dataset.shuffle(len(x_val)).batch(batch_size=trainer_config.batch_size).repeat()

        history = model.fit(train_dataset, steps_per_epoch=int(len(x_train) / trainer_config.batch_size), validation_data=val_dataset,
                            epochs=epochs, verbose=1, validation_steps=len(x_train) // trainer_config.batch_size,
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

    trainer_config.set_trained_model_path(save_dir + 'model_ep{}.h5'.format(len(history.epoch)))

    y_pred = []
    dice_scores_string = []
    dice_scores = []
    for x, y in zip(x_test, y_test):
        img_pred = np.empty((x.shape))
        dice_score_img = []
        for slice in range(x.shape[0]):
            temp = np.expand_dims(x[slice], -1)  # expand dims for channel
            temp = np.expand_dims(temp, 0)  # expand dims for batch
            prediction = model.predict(temp, verbose=0)
            prediction = np.squeeze(prediction)
            prediction_bin = np.where(prediction > 0.9, 1, 0)
            img_pred[slice, ...] = prediction_bin
            dice_scores.append(su.dice(y[slice], prediction_bin))
            dice_score_img.append('dice: ' + str(np.round(su.dice(y[slice], np.where(prediction > 0.9, 1, 0)), 4)))
        y_pred.append(img_pred)
        dice_scores_string.append(dice_score_img)

    print('mean Dice score: ', np.mean(dice_scores))
    if not os.path.exists(os.path.join(save_dir, 'testset_vis')):
        os.makedirs(os.path.join(save_dir, 'testset_vis'))
    file_names = []
    for i in range(len(y_pred)):
        masked = np.multiply(y_pred[i], x_test[i])
        x_test_affine = x_test_data[i].affine
        x_test_header = x_test_data[i].header
        file_name = os.path.basename(x_test_data[i].file_map['image'].filename)
        file_names.append(file_name)

        y_true_temp = np.moveaxis(y_test[i], 0, 2)
        y_true_temp = nib.Nifti1Image(y_true_temp, x_test_affine, x_test_header)
        nib.save(y_true_temp, os.path.join(save_dir, 'testset_vis', 'y_true_' + file_name))

        img = nib.Nifti1Image(masked, x_test_affine, x_test_header)
        nib.save(img, os.path.join(save_dir, 'testset_vis', 'masked_' + file_name))

    thresholds = [0.9]
    outputs = []
    slice_titles = [None, None, dice_scores_string, None, None, None, None]
    for thr in thresholds:
        if thr == 0:
            outputs.append([np.squeeze(img) for img in y_pred])
        else:
            outputs.append([np.where(np.squeeze(img) > thr, 1, 0) for img in y_pred])
    list = [x_test, y_test]
    for o in outputs:
        list.append(o)

    if visualisation == True:
        utils.save_datavisualisation_plt(list, os.path.join(save_dir, 'testset_vis/'), normalized=True,
                                         file_names=file_names,
                                         figure_title='Predictions with a mean Dice score of {}'.format(
                                             np.round(np.mean(dice_scores), 4)), slice_titles=slice_titles)

    np.save(save_dir + 'y_pred_{}dice'.format(np.round(np.mean(dice_scores), 4)), y_pred)

    try:
        import subprocess
        git_commit_version = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
        experiment_config['git_commit'] = git_commit_version
    except Exception as e:
        print(e)
    experiment_config['date_time'] = str(datetime.datetime.now())
    experiment_config['augmentation_params'] = data_gen_args
    experiment_config['seed'] = seed
    experiment_config['epochs'] = len(history.epoch)
    experiment_config['shape'] = shape
    experiment_config['model_path'] = save_dir + 'model_ep{}.h5'.format(len(history.epoch))
    experiment_config['dice_score'] = np.round(np.mean(dice_scores), 4)
    df = pd.DataFrame(columns=[key for key in experiment_config.keys()])
    df = df.append(experiment_config, ignore_index=True)
    df.to_csv(save_dir + 'experiment_config.csv', index=False)

    if trainer_config.model_comparison_df:
        results_df = pd.read_csv(trainer_config.model_comparison_df)
        for key in results_df.columns:
            if key not in df.columns:
                df[key] = np.nan
        for key in df.columns:
            if key not in results_df.columns:
                results_df[key] = np.nan
        results_df = results_df.append(df, ignore_index=True)
        results_df.to_csv(trainer_config.model_comparison_df, index=False)

    if not test:
        tester(data_dir, ['irsabi'], os.path.join(save_dir, 'vis', data_type + '/'),
               save_dir + 'model_ep{}.h5'.format(len(history.epoch)), data_type)
    return history


def network_trainer(trainer_config, file_name, data_dir, template_dir, test, loss, epochss, shape, data_gen_argss,
                    blacklist, data_type, slice_view, visualisation=False, pretrained_model=False, data_sets=[''],
                    excluded_from_training=['']):
    """
    This function loads the data, preprocesses it and trains the network with given parameters.
    It trains the network successively with different data augmentation values.
    If the training is early stopped before 'min_epochs', the training is started again with reduced augmentation values

    :param test: Bool: If Test is True, every parameter is set to increase learning speed. Used to test if the code runs
    :param loss: string: loss with which the model will be trained
    :param epochss: Array with epochs. Should have the same length than data_gen_argss
    :param shape: Tuple (y,x): Shape of the images that should come out of the preprocessing
    :param data_gen_argss: Array of dicts : arguments for the data augmentations, should have the same length than epochss
    :param min_epochs: int: The minimum amount of epochs the network should be trained on. If this number is not reached, the training will start again with reduced augmentation values
    :param max_tries: int: Integer indicating how many times the training should be started again with reduced augmentation values
    :param visualisation: Bool: if True, all images after preprocessing are saved
    :return: Bool: True if min_epochs is not reached, False otherwise
    """
    experiment_config = {}
    experiment_config['data_sets'] = data_sets
    experiment_config['excluded'] = excluded_from_training
    experiment_config['slice_view'] = slice_view
    experiment_config['pretrained_model'] = pretrained_model
    experiment_config['data_type'] = data_type
    experiment_config['uid'] = uuid.uuid4()
    experiment_config['loss'] = loss
    experiment_config['blacklist'] = trainer_config.blacklist_dir
    experiment_config['model'] = trainer_config.model

    seed = random.randint(0, 1000)

    print('Training with seed: ', seed)

    if data_type == 'anat':
        img_data = dl.load_img(data_dir, blacklist, studies=data_sets)
        excluded_img_data = dl.load_img(data_dir, studies=excluded_from_training)
    elif data_type == 'func':
        img_data = dl.load_func_img(data_dir, blacklist, studies=data_sets)
        excluded_img_data = dl.load_func_img(data_dir, studies=excluded_from_training)

    save_dir = data_dir + 'results/' + file_name + '/{loss}_{epochs}_{date}/'.format(loss=loss,
                                                                                     epochs=np.sum(epochss),
                                                                                     date=datetime.date.today())

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print('creating dir: ', save_dir)

    """shape = (z,y,x)"""

    temp = dl.load_mask(template_dir)
    mask_data = []
    excluded_mask_data = []

    for i in range(len(img_data)):
        mask_data.append(copy.deepcopy(temp))

    for i in range(len(excluded_img_data)):
        excluded_mask_data.append(copy.deepcopy(temp))

    print('*** Splitting data into Train, Validation and Test set ***')

    x_train1_data, x_test_data, y_train1_data, y_test_data = model_selection.train_test_split(img_data, mask_data,
                                                                                              random_state=seed,
                                                                                              test_size=0.1,
                                                                                              shuffle=True)

    print('*** Preprocessing ***')
    x_test_data.extend(excluded_img_data)
    y_test_data.extend(excluded_mask_data)
    x_test, y_test, x_test_affines, x_test_headers, file_names, y_test_affines, y_test_headers = utils.get_image_and_mask(
        x_test_data, y_test_data, shape, save_dir, slice_view=slice_view, visualisation=visualisation,
        blacklist_bool=blacklist, test = test)
    x_train1, y_train1, x_train1_affines, x_train1_headers, x_train1_file_names, = utils.get_image_and_mask(
        x_train1_data, y_train1_data, shape, save_dir, slice_view=slice_view, visualisation=visualisation,
        blacklist_bool=blacklist, test = test)[:5]

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
    x_train, x_val, y_train, y_val = model_selection.train_test_split(x_train1, y_train1, test_size=0.25, shuffle=True,
                                                                      random_state=seed)

    print('TRAINING SHAPE: ' + str(x_train.shape[1:4]))
    print('*** Training with {} slices ***'.format(x_train.shape[0]))
    print('*** Validating with {} slices ***'.format(x_val.shape[0]))
    input_shape = (x_train.shape[1:4])

    """
    Callbacks
    """

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, verbose=1, patience=5)
    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    Adam = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, amsgrad=True)

    callbacks = [reduce_lr, earlystopper]

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
        if trainer_config.model == 'attention_unet':
            model = attention_unet.att_unet(input_shape[0], input_shape[1], 1)
        elif trainer_config.model == 'unet':
            model = unet.unet(input_shape)
        elif trainer_config.model == 'att_r2_unet':
            model = attention_unet.att_r2_unet(input_shape[0], input_shape[1], 1)

    else:
        print(pretrained_model)
        model = keras.models.load_model(pretrained_model, custom_objects={'dice_coef_loss': unet.dice_coef_loss,
                                                                          'dice_coef': unet.dice_coef})

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
        else:
            augmentation = True

        if counter > 1:
            print(
                '\n\n\n\n********* \nTraining with higher augmentation values! Taking model from try {} \n*********\n\n\n\n'.format(
                    best_try + 1))
            model = history.model

        histories = []
        new_save_dir = save_dir + '{counter}_Step/'.format(counter=counter)

        if not os.path.exists(new_save_dir):
            os.makedirs(new_save_dir)
        if data_gen_args == data_gen_argss[-1]:
            last_step = True
            # callbacks[-1] = EarlyStopping(monitor='val_loss', patience=80, verbose=1)
        else:
            last_step = False

        print('Step', counter, 'of', len(epochss))
        temp_history = training(experiment_config, trainer_config, data_dir, data_sets, data_type, data_gen_args,
                                epochs, loss, shape, x_train, y_train, x_val, y_val, x_test, y_test, new_save_dir,
                                x_test_data, model, seed, Adam, callbacks, slice_view, augmentation=augmentation,
                                visualisation=visualisation, last_step=last_step, test=test)
        histories.append(temp_history)

        history_epochs = []
        for x in histories:
            history_epochs.append(len(x.epoch))
            best_try = history_epochs.index(max(history_epochs))  # best_try is the try with the most epochs
        history = histories[best_try]
        counter += 1
