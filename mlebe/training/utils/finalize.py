import json
import os

from mlebe.training.classifier_tester import evaluate


def finalize(json_opts, json_filename, model, experiment_config):
    """
    function that saves the experiment results and tests the model
    """

    # get the model path with the epoch of the best model
    model_path = os.path.join(model.save_dir, '{0:03d}_net_{1}.pth'.format(model.best_epoch,
                                                                           json_opts.model.model_type))
    # save config with path of trained model
    with open(json_filename) as file:
        config = json.load(file)
    config['model']['path_pre_trained_model'] = model_path
    config['model']['isTrain'] = False
    if experiment_config:
        config['model']['uid'] = experiment_config.experiment_config['uid'].item()
    config_trained_path = os.path.join(model.save_dir, 'trained_' + json_filename.split('/')[-1])
    with open(config_trained_path, 'w') as outfile:
        json.dump(config, outfile, indent=4)
    if experiment_config:
        experiment_config.experiment_config['model_path'] = model_path
        experiment_config.experiment_config['config_path'] = config_trained_path

    # testing the model:
    irsabi_dice_mean, irsabi_dice_std = evaluate(config_trained_path)

    return irsabi_dice_mean, irsabi_dice_std
