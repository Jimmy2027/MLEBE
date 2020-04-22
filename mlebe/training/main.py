import network_trainer
import utils
import os
"""
:param test: Bool: If Test is True, every parameter is set to increase learning speed. Used to test if the code runs
:param remote: Bool: If remote is True, the paths are set for remote computer
:param visualisation: Bool: if True, all images in the preprocessing steps are saved
:param loss: Array of strings: with which loss the network will be trained
:param epochss: Array with epochs. Should have the same length than data_gen_argss
:param min_epochs: int: The minimum amount of epochs the network should be trained on. If this number is not reached, the training will start again with reduced augmentation values
:param data_gen_argss: Array of dicts : arguments for the data augmentations, should have the same length than epochss
:param max_tries: int: Integer indicating how many times the training should be started again with reduced augmentation values
:param shape: Tuple (y,x): Shape of the images that should come out of the preprocessing
"""

file_name = 'attention_r2unet_func'
# file_name = 'test'

pretrained_model = '/mnt/data/mlebe_data/results/attention_unet/dice_600_2020-04-20/1_Step/model_ep113.h5'
data_dir = '/mnt/data/mlebe_data/' #directory of the training data
template_dir = '/usr/share/mouse-brain-atlases/' #directory of the template

"""
Parameters
"""

pretrained_model = False  #needs to be string (path to model) if True
blacklist = False
slice_view = 'coronal'
data_type = 'anat'
shape = (128, 128)  #original image shape: (63,96,48) with coronal_slice: (63,48), sagittal: (96, 48), axial: (63,96)
visualisation = False    #if visualisation true saves pre- and unpreprocessed images for visualisation
test = False
losses = ['dice']
data_sets = ['drlfom', 'mgtdbs', 'opfvta', 'ztau', 'hendrik_nvcz']
excluded_from_training = ['irsabi']
# data_sets = ['ztau']

epochss = [600]
if test == True:
    min_epochs = 0
if data_type == 'func':
    blacklist = False

data_gen_args = dict(
    rotation_range=90,
    width_shift_range=30,
    height_shift_range=30,
    shear_range=0.5,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    brightness_range = (0.7, 1.3),
    noise_var_range = (0, 0.001),    #variance range of the gaussian noise that is added to the image
    bias_var_range = (5, 10),   #width of the bias added on top of the images
    )

data_gen_argss = [data_gen_args]

if blacklist == True:
    if os.path.isdir(os.path.expanduser('~/src/MLEBE/mlebe/Blacklist')):
        blacklist = utils.general.write_blacklist(os.path.expanduser('~/src/MLEBE/mlebe/Blacklist'))
    else:
        blacklist = False
        print('No Blacklist dir found')

for i, loss in enumerate(losses):
    network_trainer.network_trainer(file_name, data_dir, template_dir, test, loss, epochss, shape, data_gen_argss, blacklist, data_type, slice_view = slice_view, visualisation=visualisation, pretrained_model = pretrained_model, data_sets = data_sets, excluded_from_training=excluded_from_training)


