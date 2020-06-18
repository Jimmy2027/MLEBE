from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from mlebe.threed.training.dataio.loaders.mlebe_loader import Experiment_config
from mlebe.threed.training.train_segmentation import train

# parameters to try: loss, augmentations, blacklist, remove_black_slices, with_elastic_transform
# todo make notebook where compare predictions on bids and preprocessed
params_seach_space_anat = {
    'criterion': ['dice_loss', 'cross_entropy'],
    'with_blacklist': [True],
    'model_type': ['unet_nonlocal', 'unet_pct_multi_att_dsv', 'unet'],
    'normalization': ['normalize_medic'],
    'random_elastic_prob': [0],
    'with_arranged_mask': [True],
    "scale_size": [[64, 64, 96, 1]],
    "bias_field_prob": [0.5]
}
params_seach_space_func = {
    'criterion': ['dice_loss', 'cross_entropy'],
    'model_type': ['unet_nonlocal', 'unet_pct_multi_att_dsv', 'unet'],
    'with_blacklist': [False],
    'normalization': ['normalize_medic'],
    'random_elastic_prob': [0],
    'with_arranged_mask': [True],
    "scale_size": [[64, 64, 96, 1]],
    "bias_field_prob": [0.5]
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
        experiment_config.check_if_already_tried()
        if not experiment_config.already_tried:
            # train model
            results, model_path = train(config_path, params=params)
            # remove all "class" scores
            results = results.loc[:, ~results.columns.str.contains('^Class')]
            experiment_config.experiment_config['model_path'] = model_path
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
