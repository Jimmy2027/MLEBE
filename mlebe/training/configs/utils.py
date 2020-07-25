import json
import os
import pandas as pd


def write_to_jsonfile(config_path: str, parameters: list):
    """
    parameters: list of tuples. Example [('model.use_cuda',VALUE),] where VALUE is the parameter to be set
    """
    with open(config_path) as file:
        config = json.load(file)
    for parameter, value in parameters:
        split = parameter.split('.')
        idx = 1
        key = config[split[0]]
        while idx < len(split) - 1:
            key = key[split[idx]]
            idx += 1

        key[split[-1]] = value

    with open(config_path, 'w') as outfile:
        json.dump(config, outfile, indent=4)


def json_to_dict(json_path):
    with open(json_path) as json_file:
        dict = json.load(json_file)

    return dict


def get_dice_score_of_model(model_config_path,
                            experiment_results_df_path='/home/hendrik/src/MLEBE/mlebe/training/three_D/results.csv'):
    experiment_results_df = pd.read_csv(experiment_results_df_path)
    model_uid = json_to_dict(model_config_path)['model']['uid']
    dice_score = experiment_results_df.loc[experiment_results_df['uid'] == model_uid, 'Overall_Dice'].item()

    return dice_score


def twoD_json_config_wrapper(model_dir):
    if not os.path.exists(os.path.join(model_dir, 'config.json')):
        model_config = pd.read_csv(os.path.join(model_dir, 'experiment_config.csv')).iloc[0].to_dict()
        with open(os.path.join(model_dir, 'config.json'), 'w') as outfile:
            json.dump(model_config, outfile, skipkeys=True, indent=4)
