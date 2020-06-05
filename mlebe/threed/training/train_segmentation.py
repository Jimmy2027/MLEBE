from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from dataio.loaders import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from utils.utils import json_file_to_pyobj
from utils.visualiser import Visualiser
from utils.error_logger import ErrorLogger
import pandas as pd
import os
from models import get_model
import shutil


# todo visdom visualisation needs to be an option

def train(json_filename, network_debug=False):
    # Load options
    json_opts = json_file_to_pyobj(json_filename)
    train_opts = json_opts.training

    # Architecture type
    arch_type = train_opts.arch_type

    # Setup Dataset and Augmentation
    ds_class = get_dataset('mlebe_dataset')
    data_type = json_opts.data.data_type
    ds_path = json_opts.data.data_dir
    template_path = json_opts.data.template_dir
    studies = json_opts.data.studies
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
    if json_filename == 'configs/temp_config.json':
        print('removing dir ', model.save_dir)
        shutil.rmtree(model.save_dir)
        os.mkdir(model.save_dir)

    if network_debug:
        print('# of pars: ', model.get_number_parameters())
        print('fp time: {0:.3f} sec\tbp time: {1:.3f} sec per sample'.format(*model.get_fp_bp_time()))
        exit()

    # Setup Data Loader
    split_opts = json_opts.data_split
    train_dataset = ds_class(template_path, ds_path, studies, split='train', save_dir=model.save_dir,
                             data_type=data_type,
                             transform=ds_transform['train'],
                             train_size=split_opts.train_size, test_size=split_opts.test_size,
                             valid_size=split_opts.validation_size, split_seed=split_opts.seed)
    valid_dataset = ds_class(template_path, ds_path, studies, split='validation', save_dir=model.save_dir,
                             data_type=data_type,
                             transform=ds_transform['valid'],
                             train_size=split_opts.train_size, test_size=split_opts.test_size,
                             valid_size=split_opts.validation_size, split_seed=split_opts.seed)
    test_dataset = ds_class(template_path, ds_path, studies, split='test', save_dir=model.save_dir, data_type=data_type,
                            transform=ds_transform['valid'],
                            train_size=split_opts.train_size, test_size=split_opts.test_size,
                            valid_size=split_opts.validation_size, split_seed=split_opts.seed,
                            excluded_from_training=json_opts.data.excluded_from_training)
    train_loader = DataLoader(dataset=train_dataset, num_workers=16, batch_size=train_opts.batchSize, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, num_workers=16, batch_size=train_opts.batchSize, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, num_workers=16, batch_size=train_opts.batchSize, shuffle=False)

    # Visualisation Parameters
    visualizer = Visualiser(json_opts.visualisation, save_dir=model.save_dir)
    error_logger = ErrorLogger()

    # Training Function
    model.set_scheduler(train_opts)

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
                    volumes = model.get_current_volumes()
                    visualizer.display_current_volumes(volumes, ids, split, epoch)
                    validation_volumes.append(volumes)

        # Update the plots
        for split in ['train', 'validation', 'test']:
            visualizer.plot_current_errors(epoch, error_logger.get_errors(split), split_name=split)
            visualizer.print_current_errors(epoch, error_logger.get_errors(split), split_name=split)
        visualizer.save_plots(epoch, save_frequency=5)
        current_loss = error_logger.get_errors('validation')['Seg_Loss']
        error_logger.reset()

        # Save the model parameters

        val_loss_log = pd.read_excel(os.path.join('checkpoints', json_opts.model.experiment_name, 'loss_log.xlsx'),
                                     sheet_name='validation').iloc[:, 1:]
        best_loss = val_loss_log['Seg_Loss'].min()
        if epoch % train_opts.save_epoch_freq == 0:
            model.save(json_opts.model.model_type, epoch)

        # Update the model learning rate
        model.update_learning_rate()

        if current_loss <= best_loss or epoch < 100:
            idx_early_stopping = 0
            print('current loss {} improved from {}'.format(current_loss, best_loss),
                  '-- idx_early_stopping = {} / {}'.format(idx_early_stopping,
                                                           json_opts.training.early_stopping_patience))
        else:
            idx_early_stopping += 1
            print('current loss {} did not improve from {}'.format(current_loss, best_loss),
                  '-- idx_early_stopping = {} / {}'.format(idx_early_stopping,
                                                           json_opts.training.early_stopping_patience))

        if idx_early_stopping >= json_opts.training.early_stopping_patience:
            print('early stopping')
            model_path = os.path.join(model.save_dir, '{0:03d}_net_{1}.pth'.format(
                val_loss_log.loc[val_loss_log['Seg_Loss'] == best_loss, 'epoch'].item(),
                json_opts.model.model_type))
            command = 'cp {} {}'.format(json_filename, os.path.join(model.save_dir, json_filename.split('/')[-1]))
            print(command)
            os.system(command)

            return val_loss_log.loc[val_loss_log['Seg_Loss'] == best_loss], model_path

    # get the model path with the epoch of the best model
    model_path = os.path.join(model.save_dir, '{0:03d}_net_{1}.pth'.format(
        val_loss_log.loc[val_loss_log['Seg_Loss'] == best_loss, 'epoch'].item(),
        json_opts.model.model_type))

    # save config with path of trained model
    with open(json_filename) as file:
        config = json.load(file)
    config['model']['path_pre_trained_model'] = model_path
    config['model']['isTrain'] = False
    with open(os.path.join(model.save_dir, json_filename.split('/')[-1])) as outfile:
        json.dump(config, outfile)

    return val_loss_log.loc[val_loss_log['Seg_Loss'] == best_loss], model_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CNN Seg Training Function')

    parser.add_argument('-c', '--config', help='training config file', required=True)
    parser.add_argument('-d', '--debug', help='returns number of parameters and bp/fp runtime', action='store_true')
    args = parser.parse_args()

    train(args)
