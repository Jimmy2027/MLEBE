from mlebe.threed.masking.predict_mask import predict_mask
from mlebe.threed.training.utils.utils import write_to_config
from mlebe.training.utils import data_loader
import os
from tqdm import tqdm
from mlebe.threed.training.models import get_model
from mlebe.threed.training.utils.utils import json_file_to_pyobj

"""
With this script one can visualize the performance of the masking functions for a given data set
data_paths: paths to the data that is to be used for testing
"""


def tester(data_dir, studies, save_dir, model_path, data_type='anat', visualisation_format='pdf', bias_correct=True):
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
                             bias_correct_bool=bias_correct, model_path=model_path,
                             visualisation_format=visualisation_format)[
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
                                       visualisation_path=visualisation['path'], bias_correct_bool=bias_correct,
                                       model_path=model_path, visualisation_format=visualisation_format)[0]
            command = 'mv {a} {b}'.format(a=masked_path, b=os.path.join(save_dir, 'masked_' + os.path.basename(path)))
            print(command)
            os.system(command)

        command = 'rm *.nii.gz'
        os.system(command)


if __name__ == '__main__':
    data_dir = os.path.expanduser('/usr/share/')
    studies = ['irsabi_bidsdata']
    anat_model_path = '/home/hendrik/src/MLEBE/mlebe/threed/training/checkpoints/bias_field_prob-0.5_criterion-cross_entropy_model_type-unet_pct_multi_att_dsv_normalization-normalize_medic_random_elastic_prob-0_scale_size-[64, 64, 96, 1]_with_arranged_mask-True_with_blacklist-True1/trained_mlebe_config_anat.json'
    write_to_config(anat_model_path, [('model.use_cuda', False)])
    func_model_path = '/home/hendrik/src/MLEBE/mlebe/threed/training/checkpoints/bias_field_prob-0.5_criterion-dice_loss_model_type-unet_pct_multi_att_dsv_normalization-normalize_medic_random_elastic_prob-0_scale_range-[-10, 20]_scale_size-[64, 64, 96, 1]_with_arranged_mask-True_with_blacklist-False1/trained_mlebe_config_func.json'

    tester(data_dir, studies, save_dir='visualisation/anat_with_cropping_woutbiascor/', model_path=anat_model_path,
           data_type='anat', bias_correct=False)
    tester(data_dir, studies, save_dir='visualisation/func_with_cropping_wbiascor/', model_path=func_model_path, data_type='func', bias_correct=True)
    tester(data_dir, studies, save_dir='visualisation/func_no_biascorrect_with_cropping/', model_path=func_model_path, data_type='func', bias_correct=False)
