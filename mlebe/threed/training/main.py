from mlebe.threed.training.dataio.loaders.mlebe_loader import experiment_config
from mlebe.threed.training.train_segmentation import train
from mlebe.threed.training.utils.utils import json_file_to_pyobj
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import json
from datetime import date
import os

# todo make function that can change some parameters of config file for grid search
# parameters to try: loss, augmentations, blacklist, remove_black_slices
params_seach_space_anat = {
    'criterion': ['focal_tversky_loss', 'dice_loss'],
    'with_blacklist': [True, False],
}
params_seach_space_func = {
    'criterion': ['focal_tversky_loss', 'dice_loss'],
    'with_blacklist': [False],
}

config_paths = ['configs/temp_config.json']
# config_paths = ['configs/mlebe_config_func.json']
for config_path in config_paths:
    if config_path == 'configs/mlebe_config_func.json':
        params_seach_space = params_seach_space_func
    else:
        params_seach_space = params_seach_space_anat

    for params in tqdm(list(ParameterGrid(params_seach_space))):

        with open(config_path) as file:
            config = json.load(file)
        config['model']['criterion'] = params['criterion']
        config['data']['with_blacklist'] = params['with_blacklist']
        config['model']['experiment_name'] = str(date.today()) + '_' + config['data']['data_type'] + '_' + \
                                             config['model'][
                                                 'criterion']
        with open(config_path, 'w') as outfile:
            json.dump(config, outfile, indent=4)
        # train model
        results, model_path = train(config_path, params=params)
        # remove all "class" scores
        results = results.loc[:, ~results.columns.str.contains('^Class')]
        json_opts = json_file_to_pyobj(config_path)
        exp_config_class = experiment_config(json_opts)
        exp_config_class.experiment_config['model_path'] = model_path
        # todo why is this not working? this concatenates the dataframes on axis = 0 while I want to concatenate them on axis = 1
        # exp_config_class.experiment_config = pd.concat([results, exp_config_class.experiment_config], axis=1)
        for key, value in zip(results.columns, results.values[0]):
            exp_config_class.experiment_config[key] = value

        if 'temp_config' in config_path:
            exp_config_class.save('temp_results')
        else:
            exp_config_class.save()
