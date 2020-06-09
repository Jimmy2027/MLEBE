import json
import os
from mlebe.threed.training.model_tester import tester
from mlebe.threed.training.utils.utils import json_file_to_pyobj


def finalize(json_opts, json_filename, model, val_loss_log, best_loss, test_dataset):
    """
    function that saves the experiment results and tests the model
    """

    # get the model path with the epoch of the best model
    model_path = os.path.join(model.save_dir, '{0:03d}_net_{1}.pth'.format(
        val_loss_log.loc[val_loss_log['Seg_Loss'] == best_loss, 'epoch'].item(),
        json_opts.model.model_type))

    # save config with path of trained model
    with open(json_filename) as file:
        config = json.load(file)
    config['model']['path_pre_trained_model'] = model_path
    config['model']['isTrain'] = False
    config_trained_path = os.path.join(model.save_dir, 'trained_' + json_filename.split('/')[-1])
    with open(config_trained_path, 'w') as outfile:
        json.dump(config, outfile, indent=4)

    # testing the model:
    # tester(json_file_to_pyobj(config_trained_path), test_dataset, save_directory=os.path.join(model.save_dir, 'test'))

    return model_path
