import os
import shutil

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from mlebe.training.dataio.loaders import get_dataset
from mlebe.training.dataio.transformation import get_dataset_transformation
from mlebe.training.models import get_model
from mlebe.training.utils.finalize import finalize
from mlebe.training.utils.error_logger import ErrorLogger
from mlebe.training.utils.utils import json_file_to_pyobj, bigprint
from mlebe.training.utils.visualiser import Visualiser
from mlebe.training.models.utils import EarlyStopper


def train(json_filename, network_debug=False, experiment_config=None):
    """
Main training function for the model.
    Parameters
    ----------
    json_filename : str
        Path to the json configuration file
    network_debug : bool (optional)
    experiment_config : class used for logging (optional)
    """

    # Load options
    json_opts = json_file_to_pyobj(json_filename)
    bigprint(f'New try with parameters: {json_opts}')
    train_opts = json_opts.training

    # Setup Dataset and Augmentation
    ds_class = get_dataset('mlebe_dataset')
    ds_path = json_opts.data.data_dir
    template_path = json_opts.data.template_dir
    ds_transform = get_dataset_transformation('mlebe', opts=json_opts.augmentation,
                                              max_output_channels=json_opts.model.output_nc)

    # Setup channels
    channels = json_opts.data_opts.channels
    if len(channels) != json_opts.model.input_nc \
            or len(channels) != getattr(json_opts.augmentation, 'mlebe').scale_size[-1]:
        raise Exception(
            'Number of data channels must match number of model channels, and patch and scale size dimensions')

    # Setup the NN Model
    model = get_model(json_opts.model)
    if json_filename == 'configs/test_config.json':
        print('removing dir ', model.save_dir)
        shutil.rmtree(model.save_dir)
        os.mkdir(model.save_dir)

    if network_debug:
        print('# of pars: ', model.get_number_parameters())
        print('fp time: {0:.3f} sec\tbp time: {1:.3f} sec per sample'.format(*model.get_fp_bp_time()))
        exit()

    # Setup Data Loader
    split_opts = json_opts.data_split
    data_opts = json_opts.data
    train_dataset = ds_class(template_path, ds_path, data_opts, split='train', save_dir=model.save_dir,
                             transform=ds_transform['train'],
                             train_size=split_opts.train_size, test_size=split_opts.test_size,
                             valid_size=split_opts.validation_size, split_seed=split_opts.seed,
                             training_shape=json_opts.augmentation.mlebe.scale_size[:3])
    valid_dataset = ds_class(template_path, ds_path, data_opts, split='validation', save_dir=model.save_dir,
                             transform=ds_transform['valid'],
                             train_size=split_opts.train_size, test_size=split_opts.test_size,
                             valid_size=split_opts.validation_size, split_seed=split_opts.seed,
                             training_shape=json_opts.augmentation.mlebe.scale_size[:3])
    test_dataset = ds_class(template_path, ds_path, data_opts, split='test', save_dir=model.save_dir,
                            transform=ds_transform['valid'],
                            train_size=split_opts.train_size, test_size=split_opts.test_size,
                            valid_size=split_opts.validation_size, split_seed=split_opts.seed,
                            training_shape=json_opts.augmentation.mlebe.scale_size[:3])
    train_loader = DataLoader(dataset=train_dataset, num_workers=1, batch_size=train_opts.batchSize, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, num_workers=1, batch_size=train_opts.batchSize, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, num_workers=1, batch_size=train_opts.batchSize, shuffle=False)

    # Visualisation Parameters
    visualizer = Visualiser(json_opts.visualisation, save_dir=model.save_dir)
    error_logger = ErrorLogger()

    # Training Function
    model.set_scheduler(train_opts)
    # Setup Early Stopping
    early_stopper = EarlyStopper(json_opts.training.early_stopping_patience)

    for epoch in range(model.which_epoch, train_opts.n_epochs):
        print('(epoch: %d, total # iters: %d)' % (epoch, len(train_loader)))
        train_volumes = []
        validation_volumes = []

        # Training Iterations
        for epoch_iter, (images, labels, indices) in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
            # Make a training update
            model.set_input(images, labels)
            model.optimize_parameters()
            # model.optimize_parameters_accumulate_grd(epoch_iter)

            # Error visualisation
            errors = model.get_current_errors()
            error_logger.update(errors, split='train')

            ids = train_dataset.get_ids(indices)
            volumes = model.get_current_volumes()
            visualizer.display_current_volumes(volumes, ids, 'train', epoch)
            train_volumes.append(volumes)

        # Validation and Testing Iterations
        for loader, split, dataset in zip([valid_loader, test_loader], ['validation', 'test'],
                                          [valid_dataset, test_dataset]):
            for epoch_iter, (images, labels, indices) in tqdm(enumerate(loader, 1), total=len(loader)):
                ids = dataset.get_ids(indices)

                # Make a forward pass with the model
                model.set_input(images, labels)
                model.validate()

                # Error visualisation
                errors = model.get_current_errors()
                stats = model.get_segmentation_stats()
                error_logger.update({**errors, **stats}, split=split)

                # Visualise predictions
                if split == 'validation':  # do not look at testing
                    # Visualise predictions
                    volumes = model.get_current_volumes()
                    visualizer.display_current_volumes(volumes, ids, split, epoch)
                    validation_volumes.append(volumes)

        current_loss = error_logger.get_errors('validation')['Seg_Loss']
        # Update best validation loss/epoch values
        model.update_validation_state(epoch, current_loss)
        early_stopper.update(model, epoch, current_loss)
        # Update the plots
        for split in ['train', 'validation', 'test']:
            visualizer.plot_current_errors(epoch, error_logger.get_errors(split), split_name=split)
            visualizer.print_current_errors(epoch, error_logger.get_errors(split), split_name=split)
        visualizer.save_plots(epoch, save_frequency=5)
        error_logger.reset()

        # saving checkpoint
        if model.is_improving:
            print('saving model')
            # replacing old model with new model
            model.save(json_opts.model.model_type, epoch)

        # Update the model learning rate
        model.update_learning_rate(metric=current_loss)

        if early_stopper.should_stop_early:
            print('early stopping')
            # get validation metrics
            val_loss_log = pd.read_excel(os.path.join('checkpoints', json_opts.model.experiment_name, 'loss_log.xlsx'),
                                         sheet_name='validation').iloc[:, 1:]

            irsabi_dice_mean, irsabi_dice_std = finalize(json_opts, json_filename, model, experiment_config)

            val_loss_log['irsabi_dice_mean'] = irsabi_dice_mean
            val_loss_log['irsabi_dice_std'] = irsabi_dice_std
            return val_loss_log.loc[val_loss_log['Seg_Loss'] == val_loss_log['Seg_Loss'].min()]

    # get validation metrics
    val_loss_log = pd.read_excel(
        os.path.join(json_opts.model.checkpoints_dir, json_opts.model.experiment_name, 'loss_log.xlsx'),
        sheet_name='validation').iloc[:, 1:]

    irsabi_dice_mean, irsabi_dice_std = finalize(json_opts, json_filename, model, experiment_config)

    val_loss_log['irsabi_dice_mean'] = irsabi_dice_mean
    val_loss_log['irsabi_dice_std'] = irsabi_dice_std
    return val_loss_log.loc[val_loss_log['Seg_Loss'] == val_loss_log['Seg_Loss'].min()]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CNN Seg Training Function')

    parser.add_argument('-c', '--config', help='training config file', required=True)
    parser.add_argument('-d', '--debug', help='returns number of parameters and bp/fp runtime', action='store_true')
    args = parser.parse_args()

    train(args)
