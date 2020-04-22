from mlebe.training.trainer_config import trainer_config
import os

pretrained_model = '/mnt/data/mlebe_data/results/attention_unet/dice_600_2020-04-20/1_Step/model_ep113.h5'
data_dir = '/mnt/data/mlebe_data/' #directory of the training data
template_dir = '/usr/share/mouse-brain-atlases/' #directory of the template

"""
Parameters
"""
blacklist = False
slice_view = 'coronal'
shape = (128, 128)  #original image shape: (63,96,48) with coronal_slice: (63,48), sagittal: (96, 48), axial: (63,96)

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
    brightness_range = (0.7, 1.1),
    noise_var_range = (0, 0.01),    #variance range of the gaussian noise that is added to the image
    bias_var_range = None,   #width of the bias added on top of the images
    )


file_name = 'attention_unet_anat'
anat_trainer1 = trainer_config(file_name, data_dir, template_dir, slice_view, shape, loss, data_sets,excluded_from_training, epochss, [data_gen_args], blacklist = True)
anat_trainer1.train()
pretrained_model = '/mnt/data/mlebe_data/results/attention_unet_anat/dice_600_2020-04-21/1_Step/model_ep116.h5'
file_name = 'attention_unet_func'
func_trainer1 = trainer_config(file_name, data_dir, template_dir, slice_view, shape, loss, data_sets,excluded_from_training, epochss, [data_gen_args],pretrained_model= pretrained_model, data_type = 'func')
func_trainer1.train()
file_name = 'anat_no_augment'
data_gen_args = None
anat_trainer = trainer_config(file_name, data_dir, template_dir, slice_view, shape, loss, data_sets,excluded_from_training, epochss, [data_gen_args], blacklist = True)
anat_trainer.train()
file_name = 'func_no_augment'
pretrained_model = anat_trainer.trained_model_path
func_trainer = trainer_config(file_name, data_dir, template_dir, slice_view, shape, loss, data_sets,excluded_from_training, epochss, [data_gen_args],pretrained_model= pretrained_model, data_type = 'func')
func_trainer.train()


