import json
from datetime import date
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from mlebe.threed.training.dataio.loaders.mlebe_loader import experiment_config
from mlebe.threed.training.train_segmentation import train
from mlebe.threed.training.utils.utils import json_file_to_pyobj
from mlebe.threed.training.utils.set_remote_paths import set_epfl_paths
from uuid import uuid4
from mlebe.threed.training.utils.utils import make_unique_experiment_name, bigprint

# parameters to try: loss, augmentations, blacklist, remove_black_slices, with_elastic_transform
params_seach_space_anat = {
    'criterion': ['dice_loss'],
    'with_blacklist': [True],
    'normalization': ['normalize_medic'],
    'random_elastic_prob': [0, 0.5],
    'with_arranged_mask': [False, True],
}
params_seach_space_func = {
    'criterion': ['dice_loss'],
    'with_blacklist': [False],
    'normalization': ['normalize_medic'],
    'random_elastic_prob': [0, 0.5],
    'with_arranged_mask': [False, True],

}

config_paths = ['configs/mlebe_config_anat.json', 'configs/mlebe_config_func.json']
# config_paths = ['configs/test_config.json']
for config_path in config_paths:
    set_epfl_paths(config_path)
    if config_path == 'configs/mlebe_config_func.json':
        params_seach_space = params_seach_space_func
    else:
        params_seach_space = params_seach_space_anat

    for params in tqdm(list(ParameterGrid(params_seach_space))):
        exp_config_class = experiment_config(config_path)
        experiment_config.write_struct_to_config(params)
        # train model
        results, model_path = train(config_path, params=params)
        # remove all "class" scores
        results = results.loc[:, ~results.columns.str.contains('^Class')]
        exp_config_class.experiment_config['model_path'] = model_path
        # todo why is this not working? this concatenates the dataframes on axis = 0 while I want to concatenate them on axis = 1
        # exp_config_class.experiment_config = pd.concat([results, exp_config_class.experiment_config], axis=1)
        for key, value in zip(results.columns, results.values[0]):
            exp_config_class.experiment_config[key] = value

        if 'test_config' in config_path:
            exp_config_class.save('test_results')
        else:
            exp_config_class.save()
