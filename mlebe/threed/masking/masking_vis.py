from mlebe.threed.masking.predict_mask import predict_mask
from mlebe.training import data_loader
import os
from tqdm import tqdm

"""
With this script one can visualize the performance of the masking functions for a given data set
data_paths: paths to the data that is to be used for testing
"""


def tester(data_dir, studies, save_dir, model_path, data_type='anat', visualisation_format='pdf'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    visualisation = {
        'bool': True,
        'path': save_dir,
    }

    if data_type == 'anat':
        data_paths = data_loader.load_bidsdata(data_dir, studies=studies)
        for path in tqdm(data_paths):
            print(path)
            masked_path = \
                predict_mask(path, visualisation_bool=visualisation['bool'], visualisation_path=visualisation['path'],
                             bias_correct_bool=True, model_path=model_path, visualisation_format=visualisation_format)[
                    0]
            command = 'mv {a} {b}'.format(a=masked_path, b=os.path.join(save_dir, 'masked_' + os.path.basename(path)))
            print(command)
            os.system(command)

    if data_type == 'func':
        data_paths = data_loader.load_bidsdata(data_dir, studies=['irsabi_bidsdata'], input_type='func')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for path in tqdm(data_paths):
            print(path)
            masked_path = predict_mask(path, input_type='func', visualisation_bool=visualisation['bool'],
                                       visualisation_path=visualisation['path'], bias_correct_bool=True,
                                       model_path=model_path, visualisation_format=visualisation_format)[0]
            command = 'mv {a} {b}'.format(a=masked_path, b=os.path.join(save_dir, 'masked_' + os.path.basename(path)))
            print(command)
            os.system(command)

        command = 'rm *.nii.gz'
        os.system(command)


if __name__ == '__main__':
    data_dir = os.path.expanduser('/usr/share/')
    studies = ['irsabi_bidsdata']
    anat_model_path = 'configs/load_pretrained_mlebe_config.json'
    func_model_path = '/home/hendrik/src/MLEBE/mlebe/threed/training/checkpoints/with_augmentations_func/trained_mlebe_config_func.json'

    # tester(data_dir, studies, save_dir='visualisation/anat/', model_path=anat_model_path, data_type='anat')
    tester(data_dir, studies, save_dir='visualisation/func/', model_path=func_model_path, data_type='func')
