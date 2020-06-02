from mlebe.training.trainer_config import trainer_config
import os

data_dir = '/mnt/data/hendrik/mlebe_data/'  # directory of the training data
template_dir = '/usr/share/mouse-brain-atlases/'  # directory of the template
model_comparison_df = '/home/hendrik/docsrc/mlebe/prepare/classifier/results_df.csv'

"""
Parameters
"""
slice_view = 'coronal'
shape = (128, 128, 128)  # original image shape: (63,96,48) with coronal_slice: (63,48), sagittal: (96, 48), axial: (63,96)

loss = 'dice'
data_sets = ['drlfom', 'mgtdbs', 'opfvta', 'ztau', 'hendrik_nvcz', 'irsabi_dargcc']
excluded_from_training = ['irsabi']

epochss = [600]

blacklist_dir = os.path.expanduser('~/docsrc/mlebe/data/Blacklist')

data_gen_args = dict(
    rotation_range=90,
    width_shift_range=30,
    height_shift_range=30,
    shear_range=4,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    brightness_range=(0.7, 1.1),
    noise_var_range=(0, 0.01),  # variance range of the gaussian noise that is added to the image
    bias_var_range=(5, 10),  # width of the bias added on top of the images
    bias_prob=0.1
)

file_name = 'temp'
anat_trainer1 = trainer_config(file_name, data_dir, template_dir, slice_view, shape, loss, epochss, [data_gen_args],
                               blacklist=True, excluded_from_training=excluded_from_training, data_sets=data_sets,
                               model_comparison_df=model_comparison_df, model='unet',
                               blacklist_dir=blacklist_dir)
anat_trainer1.train()
pretrained_model = '/mnt/data/mlebe_data/results/att_unet_with_bl_anat/dice_600_2020-05-01/1_Step/model_ep66.h5'
file_name = 'unet_with_dargcc_func'
func_trainer1 = trainer_config(file_name, data_dir, template_dir, slice_view, shape, loss, epochss, [data_gen_args],
                               blacklist=False, pretrained_model=pretrained_model, data_type='func',
                               data_sets=data_sets, excluded_from_training=excluded_from_training,
                               model_comparison_df=model_comparison_df, model='unet')
func_trainer1.train()
