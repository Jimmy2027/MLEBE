from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from mlebe.threed.training.dataio.loaders.mlebe_loader import Experiment_config
from mlebe.threed.training.train_segmentation import train

# parameters to try: loss, augmentations, blacklist, remove_black_slices, with_elastic_transform
# todo try loss that penalizes strong change in contour of mask
params_seach_space_anat = {
    'criterion': ['dice_boundary_loss', 'dice_and_hd', ],
    'with_blacklist': [True],
    'model_type': ['unet_pct_multi_att_dsv'],
    'normalization': ['normalize_medic'],
    'with_arranged_mask': [True],
    "scale_size": [[128, 128, 96, 1]],
    "bias_field_prob": [0.5],
    "scale_range": [[0.7, 1.2]],
}
params_seach_space_func = {
    'criterion': ['dice_loss'],
    'model_type': ['unet_pct_multi_att_dsv'],
    'with_blacklist': [False],
    'normalization': ['normalize_medic'],
    'with_arranged_mask': [True],
    "scale_size": [[128, 128, 96, 1]],
    "bias_field_prob": [0.5],
    "scale_range": [[0.7, 1.2]],
}

config_paths = ['configs/test_config.json', 'configs/mlebe_config_anat.json']
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
            results = train(config_path, params=params, experiment_config=experiment_config)
            # remove all "class" scores
            results = results.loc[:, ~results.columns.str.contains('^Class')]
            # todo why is this not working? this concatenates the dataframes on axis = 0 while I want to concatenate them on axis = 1
            # exp_config_class.experiment_config = pd.concat([results, exp_config_class.experiment_config], axis=1)
            for key, value in zip(results.columns, results.values[0]):
                experiment_config.experiment_config[key] = value

            if 'test_config' in config_path:
                experiment_config.save('test_results')
            else:
                experiment_config.save()
        else:
            print('already tried this parameter combination')