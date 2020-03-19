from predict_mask import predict_mask
from mlebe.training import data_loader
import nibabel as nib
import os
"""
With this script one can visualize the performance of the masking functions for a given data set
"""


data_dir = os.path.expanduser('/usr/share/')

data_paths = data_loader.load_bidsdata(data_dir, studies = ['irsabi_bidsdata'])
save_dir = 'vis/anat/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

command = 'cp tester.py {}tester.txt'.format(save_dir)
os.system(command)

visualisation = {
    'bool': True,
    'path': save_dir,
}

for path in data_paths:
    print(path)
    masked_path = predict_mask(path, visualisation_bool = visualisation['bool'], visualisation_path = visualisation['path'], bias_correct_bool = True, model_path = '/mnt/data/mlebe_data/results/new_bl128/dice_600_2020-02-07/1_Step/model_ep483.h5')[0]
    command = 'mv {a} {b}'.format(a = masked_path, b = os.path.join(save_dir, 'masked_' + os.path.basename(path)))
    print(command)
    os.system(command)



data_paths = data_loader.load_bidsdata(data_dir, studies = ['irsabi_bidsdata'], input_type = 'func')
save_dir = 'vis/func/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
command = 'cp tester.py {}tester.txt'.format(save_dir)
os.system(command)
visualisation = {
    'bool': True,
    'path': save_dir,
}
for path in data_paths:
    print(path)
    masked_path = predict_mask(path, input_type = 'func', visualisation_bool = visualisation['bool'], visualisation_path = visualisation['path'], bias_correct_bool = True, model_path = '/mnt/data/mlebe_data/results/func_training/dice_600_2020-02-21/1_Step/model_ep86.h5')[0]
    command = 'mv {a} {b}'.format(a = masked_path, b = os.path.join(save_dir, 'masked_' + os.path.basename(path)))
    print(command)
    os.system(command)

command = 'rm *.nii.gz'
os.system(command)