import os

from mlebe.training.two_D.trainer_config import trainer_config

pretrained_model = '/mnt/data/mlebe_data/results/attention_unet/dice_600_2020-04-20/1_Step/model_ep113.h5'
data_dir = '/mnt/data/mlebe_data/'  # directory of the training data
template_dir = '/usr/share/mouse-brain-atlases/'  # directory of the template
model_comparison_df = '/home/hendrik/docsrc/mlebe/prepare/classifier/results_df.csv'

"""
Parameters
"""
slice_view = 'coronal'
shape = (128, 128)  # original image shape: (63,96,48) with coronal_slice: (63,48), sagittal: (96, 48), axial: (63,96)

loss = 'dice'
data_sets = ['drlfom', 'mgtdbs', 'opfvta', 'ztau', 'hendrik_nvcz']
excluded_from_training = ['irsabi']

epochss = [600]

blacklist_dir = os.path.expanduser('~/src/MLEBE/mlebe/Blacklist')
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
)

file_name = 'unet_anat_with_bias'
anat_trainer1 = trainer_config(file_name, data_dir, template_dir, slice_view, shape, loss, epochss, [data_gen_args],
                               blacklist=True, excluded_from_training=excluded_from_training, data_sets=data_sets,
                               model_comparison_df=model_comparison_df)
anat_trainer1.train()
pretrained_model = anat_trainer1.trained_model_path
file_name = 'unet_func_with_bias'
func_trainer1 = trainer_config(file_name, data_dir, template_dir, slice_view, shape, loss, epochss, [data_gen_args],
                               blacklist=False, pretrained_model=pretrained_model, data_type='func',
                               data_sets=data_sets, excluded_from_training=excluded_from_training,
                               model_comparison_df=model_comparison_df)
func_trainer1.train()

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
    bias_var_range=None,  # width of the bias added on top of the images
)
file_name = 'unet_anat_no_bias'
anat_trainer1 = trainer_config(file_name, data_dir, template_dir, slice_view, shape, loss, epochss, [data_gen_args],
                               blacklist=True, excluded_from_training=excluded_from_training, data_sets=data_sets,
                               model_comparison_df=model_comparison_df, tensorboard_bool=True)
anat_trainer1.train()
pretrained_model = anat_trainer1.trained_model_path
file_name = 'unet_func_no_bias'
func_trainer1 = trainer_config(file_name, data_dir, template_dir, slice_view, shape, loss, epochss, [data_gen_args],
                               blacklist=False, pretrained_model=pretrained_model, data_type='func',
                               data_sets=data_sets, excluded_from_training=excluded_from_training,
                               model_comparison_df=model_comparison_df, tensorboard_bool=True)
func_trainer1.train()

file_name = 'anat_no_augment'
data_gen_args = None
anat_trainer = trainer_config(file_name, data_dir, template_dir, slice_view, shape, loss, epochss, [data_gen_args],
                              blacklist=True, data_sets=data_sets, excluded_from_training=excluded_from_training,
                              tensorboard_bool=True)

anat_trainer.train()
file_name = 'func_no_augment'
pretrained_model = anat_trainer.trained_model_path
func_trainer = trainer_config(file_name, data_dir, template_dir, slice_view, shape, loss, data_sets,
                              excluded_from_training, epochss, [data_gen_args], pretrained_model=pretrained_model,
                              data_type='func', tensorboard_bool=True)
func_trainer.train()
