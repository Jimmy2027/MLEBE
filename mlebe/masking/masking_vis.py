import os

from mlebe.training.three_D import write_to_jsonfile
from mlebe.training.three_D import json_file_to_pyobj
from tqdm import tqdm
from mlebe.masking.predict_mask import predict_mask
from mlebe.training.two_D.utils import data_loader

"""
With this script one can visualize the performance of the masking functions for a given data set
data_paths: paths to the data that is to be used for testing
"""

#todo check if use cuda is set when predicting mask
def tester(data_dir, studies, save_dir, data_type='anat', visualisation_format='pdf', bias_correct=True,
           with_bids_cropping=False, model_config_path=''):
    config_path = 'visualisation/vis_config.json'
    config = json_file_to_pyobj(config_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if data_type == 'anat':
        write_to_jsonfile(config_path, [('masking_config.masking_config_anat.model_config_path', model_config_path),
                                        ('masking_config.masking_config_anat.bias_correct_bool', bias_correct),
                                        ('masking_config.masking_config_anat.visualisation_format',
                                         visualisation_format),
                                        ('masking_config.masking_config_anat.visualisation_bool', True),
                                        ('masking_config.masking_config_anat.visualisation_path', save_dir),
                                        ('masking_config.masking_config_anat.with_bids_cropping', with_bids_cropping)
                                        ])
        data_paths = data_loader.load_bidsdata(data_dir, studies=studies)
        for path in tqdm(data_paths):
            print(path)
            masked_path = \
                predict_mask(path, config_path)[
                    0]
            command = 'mv {a} {b}'.format(a=masked_path, b=os.path.join(save_dir, 'masked_' + os.path.basename(path)))
            print(command)
            os.system(command)

    if data_type == 'func':
        write_to_jsonfile(config_path, [
            ('masking_config.masking_config_func.model_config_path', model_config_path),
            ('masking_config.masking_config_func.bias_correct_bool', bias_correct),
            ('masking_config.masking_config_func.visualisation_format',
             visualisation_format),
            ('masking_config.masking_config_func.visualisation_bool', True),
            ('masking_config.masking_config_func.visualisation_path', save_dir),
            ('masking_config.masking_config_func.with_bids_cropping', with_bids_cropping)
        ])
        data_paths = data_loader.load_bidsdata(data_dir, studies=['irsabi_bidsdata'], input_type='func')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for path in tqdm(data_paths):
            print(path)
            masked_path = predict_mask(path, config_path, input_type='func')[0]
            command = 'mv {a} {b}'.format(a=masked_path, b=os.path.join(save_dir, 'masked_' + os.path.basename(path)))
            print(command)
            os.system(command)

        command = 'rm *.nii.gz'
        os.system(command)


if __name__ == '__main__':
    # worst case seems to be sub-4006
    data_dir = os.path.expanduser('/usr/share/')
    studies = ['irsabi_bidsdata']
    anat_config_path = "/home/hendrik/src/MLEBE/mlebe/training/three_D/checkpoints/2020_7_1_8_58_40/trained_mlebe_config_anat.json"
    func_model_path = "/home/hendrik/src/MLEBE/mlebe/training/three_D/checkpoints/2020_7_7_9_58_4/trained_mlebe_config_func.json"

    tester(data_dir, studies, save_dir='visualisation/test/',
           model_config_path=anat_config_path,
           data_type='anat', bias_correct=False, with_bids_cropping=False)
    # tester(data_dir, studies, save_dir='visualisation/func_nobiascor_nocrop/', model_config_path=func_model_path,
    #        data_type='func', bias_correct=False, with_bids_cropping=False)
    # tester(data_dir, studies, save_dir='visualisation/func_no_biascorrect_with_cropping/', model_path=func_model_path,
    #        data_type='func', bias_correct=False)
