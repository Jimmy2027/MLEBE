from mlebe.threed.training.dataio.loaders.mlebe_loader import experiment_config
from mlebe.threed.training.train_segmentation import train
from mlebe.threed.training.utils.utils import json_file_to_pyobj

#todo make function that can change some parameters of config file for grid search
# parameters to try: loss, augmentations, blacklist

# config_paths = ['configs/temp_config.json','configs/temp_config.json']
config_paths = ['configs/mlebe_config_func.json']
for config_path in config_paths:
    json_opts = json_file_to_pyobj(config_path)
    exp_config_class = experiment_config(json_opts)

    results, model_path = train(config_path)

    results = results.loc[:, ~results.columns.str.contains('^Class')]
    exp_config_class.experiment_config['model_path'] = model_path
    # todo why is this not working? this concatenates the dataframes on axis = 0 while I want to concatenate them on axis = 1
    # exp_config_class.experiment_config = pd.concat([results, exp_config_class.experiment_config], axis=1)
    for key, value in zip(results.columns, results.values[0]):
        exp_config_class.experiment_config[key] = value

    if 'temp_config' in config_path:
        exp_config_class.save('temp_results')
    else:
        exp_config_class.save()

