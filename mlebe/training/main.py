from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from mlebe.training.dataio.loaders.mlebe_loader import Experiment_config
from mlebe.training.train_segmentation import train

# parameters to try: loss, augmentations, blacklist, remove_black_slices, with_elastic_transform
# todo add dropout?
params_seach_space_anat = {
    'criterion': ['dice_boundary_loss'],
    'with_blacklist': [True],
    'model_type': ['unet_pct_multi_att_dsv'],
    'normalization': ['normalize_medic', 'mlebe'],
    'with_arranged_mask': [False, True],
    "scale_size": [[64, 64, 96, 1]],
    "bias_field_prob": [0.5],
    "scale_range": [[0.8, 1.1]],
    "optimizer": ['adam'],
    "lr_scheduler": ['plateau'],
    "with_FLASH": [False]
}
params_seach_space_func = {
    'criterion': ['dice_boundary_loss'],
    'model_type': ['unet_pct_multi_att_dsv'],
    'with_blacklist': [False],
    'normalization': ['normalize_medic', 'mlebe'],
    'with_arranged_mask': [False, True],
    "scale_size": [[64, 64, 96, 1]],
    "bias_field_prob": [0.5],
    "scale_range": [[0.8, 1.1]],
    "optimizer": ['adam'],
    "lr_scheduler": ['plateau'],
    "with_FLASH": [False]
}

config_paths = ['configs/mlebe_config_anat.json', 'configs/mlebe_config_func.json']
# config_paths = ['configs/test_config.json']
for config_path in config_paths:
    if config_path == 'configs/mlebe_config_func.json':
        params_seach_space = params_seach_space_func
    else:
        params_seach_space = params_seach_space_anat

    for params in tqdm(list(ParameterGrid(params_seach_space))):
        print(params)
        experiment_config = Experiment_config(config_path)
        experiment_config.write_struct_to_config(params)
        experiment_config.make_experiment_config_df()
        experiment_config.check_if_already_tried()
        if not experiment_config.already_tried:
            # train model
            results = train(config_path, experiment_config=experiment_config)
            # remove all "class" scores
            results = results.loc[:, ~results.columns.str.contains('^Class')]
            for key, value in zip(results.columns, results.values[0]):
                experiment_config.experiment_config[key] = value

            if 'test_config' in config_path:
                experiment_config.save('test_results')
            else:
                experiment_config.save()
        else:
            print('already tried this parameter combination')
