import os
from tqdm import tqdm
from mlebe.threed.masking.predict_mask import predict_mask as predict_mask_3D
from mlebe.masking.predict_mask import predict_mask as predict_mask_2D
from mlebe.threed.training.configs.utils import write_to_jsonfile
from mlebe.training.utils import data_loader
from mlebe.threed.training.utils.utils import json_file_to_pyobj

"""
With this script one can visualize the performance of the masking functions for a given data set
data_paths: paths to the data that is to be used for testing
"""


def tester(data_dir, studies, save_dir, data_type='anat', visualisation_format='pdf', bias_correct=True,
           with_bids_cropping=False, model_config_path=''):
    config_path = 'visualisation/twoD_config.json'
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
            if config.workflow_config.model_type == '3D':
                masked_path = \
                    predict_mask_3D(path, config_path)[
                        0]
            elif config.workflow_config.model_type == '2D':
                masked_path = \
                    predict_mask_2D(path, config_path)[
                        0]
            command = 'mv {a} {b}'.format(a=masked_path, b=os.path.join(save_dir, 'masked_' + os.path.basename(path)))
            print(command)
            os.system(command)

    if data_type == 'func':
        write_to_jsonfile(config_path, [
            ('masking_config.masking_config_func.masking_config', model_config_path),
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
            if config.workflow_config.model_type == '3D':
                masked_path = predict_mask_3D(path, config_path, input_type='func')[0]
            elif config.workflow_config.model_type == '2D':
                masked_path = predict_mask_2D(path, config_path, input_type='func')[0]
            command = 'mv {a} {b}'.format(a=masked_path, b=os.path.join(save_dir, 'masked_' + os.path.basename(path)))
            print(command)
            os.system(command)

        command = 'rm *.nii.gz'
        os.system(command)


if __name__ == '__main__':
    # worst case seems to be sub-4006
    data_dir = os.path.expanduser('/usr/share/')
    studies = ['irsabi_bidsdata']
    anat_config_path = '/mnt/data/hendrik/results/new_bl128/dice_600_2020-02-07/1_Step/experiment_config.csv'
    func_model_path = '/home/hendrik/src/MLEBE/mlebe/threed/training/checkpoints/bias_field_prob-0.5_criterion-dice_loss_model_type-unet_pct_multi_att_dsv_normalization-normalize_medic_scale_range-[0.7, 1.2]_scale_size-[128, 128, 96, 1]_with_arranged_mask-True_with_blacklist-False2/trained_mlebe_config_func.json'
    # write_to_jsonfile(anat_config_path, [('model.use_cuda', False)])
    write_to_jsonfile(func_model_path, [('model.use_cuda', False)])

    tester(data_dir, studies, save_dir='visualisation/twoD_anat_withbiascor_withcrop/',
           model_config_path=anat_config_path,
           data_type='anat', bias_correct=True, with_bids_cropping=True)
    tester(data_dir, studies, save_dir='visualisation/twoD_anat_nobiascor_nocrop/', model_config_path=anat_config_path,
           data_type='anat', bias_correct=False, with_bids_cropping=False)
    # tester(data_dir, studies, save_dir='visualisation/func_withbiascor/', model_path=func_model_path,
    #        data_type='func', bias_correct=False)
    # tester(data_dir, studies, save_dir='visualisation/func_no_biascorrect_with_cropping/', model_path=func_model_path,
    #        data_type='func', bias_correct=False)
