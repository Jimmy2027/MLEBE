from mlebe.masking.predict_mask import predict_mask
from mlebe.training import data_loader
import nibabel as nib
import os

"""
With this script one can visualize the performance of the masking functions for a given data set
data_paths: paths to the data that is to be used for testing
"""


def tester(data_dir, studies, save_dir, model_path, data_type = 'anat'):
    data_paths = data_loader.load_bidsdata(data_dir, studies=studies)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    visualisation = {
        'bool': True,
        'path': save_dir,
    }

    if data_type == 'anat':
        for path in data_paths:
            print(path)
            masked_path = \
            predict_mask(path, visualisation_bool=visualisation['bool'], visualisation_path=visualisation['path'],
                         bias_correct_bool=True, anat_model_path=model_path)[0]
            command = 'mv {a} {b}'.format(a=masked_path, b=os.path.join(save_dir, 'masked_' + os.path.basename(path)))
            print(command)
            os.system(command)

        data_paths = data_loader.load_bidsdata(data_dir, studies=['irsabi_bidsdata'], input_type='func')
        save_dir = 'vis/func/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        visualisation = {
            'bool': True,
            'path': save_dir,
        }
    if data_type =='func':
        for path in data_paths:
            print(path)
            masked_path = predict_mask(path, input_type='func', visualisation_bool=visualisation['bool'],
                                       visualisation_path=visualisation['path'], bias_correct_bool=True,
                                       func_model_path=model_path)[0]
            command = 'mv {a} {b}'.format(a=masked_path, b=os.path.join(save_dir, 'masked_' + os.path.basename(path)))
            print(command)
            os.system(command)

        command = 'rm *.nii.gz'
        os.system(command)


if __name__ == '__main__':
    data_dir = os.path.expanduser('/usr/share/')
    studies = ['irsabi_bidsdata']
    save_dir = 'vis/anat/'
    anat_model_path = '/mnt/data/mlebe_data/results/attention_unet/dice_600_2020-04-20/1_Step/model_ep113.h5'
    func_model_path = '/mnt/data/mlebe_data/results/anat_w_bias_+nvcz_/dice_600_2020-04-13/1_Step/model_ep388.h5'

    tester(data_dir, studies, save_dir, anat_model_path, data_type='anat')
    # tester(data_dir, studies, save_dir, func_model_path, data_type='func')