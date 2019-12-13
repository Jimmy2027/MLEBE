import network_trainer
import utils
import shutil
import tensorflow.keras
import os

"""
:param test: Bool: If Test is True, every parameter is set to increase learning speed. Used to test if the code runs
:param remote: Bool: If remote is True, the paths are set for remote computer
:param visualisation: Bool: if True, all images after preprocessing are saved
:param loss: Array of strings: with which loss the network will be trained
:param epochss: Array with epochs. Should have the same length than data_gen_argss
:param min_epochs: int: The minimum amount of epochs the network should be trained on. If this number is not reached, the training will start again with a different seed and reduced augmentation values
:param data_gen_argss: Array of dicts : arguments for the data augmentations, should have the same length than epochss
:param max_tries: int: Integer indicating how many times the training should be started again with reduced augmentation values
:param shape: Tuple (y,x): Shape of the images that should come out of the preprocessing


"""
if os.path.exists('/usr/share/mouse-brain-atlases/'):
    remote = True
else: remote = False

file_name = 'new_new_hope'
i = 0
while os.path.exists(file_name + '{}/'.format(i)):
    i += 1
file_name = '{filename}{i}'.format(filename = file_name, i=i)

test = True
visualisation = False  #if visualisation true saves pre- and unpreprocessed images for visualisation
remove_black_labels_and_columns = False
# losses = ['dice_bincross', 'dice', 'bincross']
losses = ['dice']
epochss = [300, 300, 300, 600]
min_epochs = 60
if test == True:
    min_epochs = 0

data_gen_args3 = dict(rotation_range=90,
                     # brightness_range=[0.9, 1.1],
                     width_shift_range=30,
                     height_shift_range=30,
                     shear_range=5,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='nearest')

data_gen_args2 = dict(rotation_range=45,
                     # brightness_range=[0.9, 1.1],
                     width_shift_range=15,
                     height_shift_range=15,
                     shear_range=5,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='nearest')

data_gen_args1 = dict(rotation_range=0.2,
                    # brightness_range=[0.9, 1.1], #if training step 1 doesn0t work anymore it's because I added this line
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest')

data_gen_args0 = None


data_gen_argss = [data_gen_args0, data_gen_args1, data_gen_args2, data_gen_args3]
max_tries = 3
if test == True:
    max_tries = 0
    # data_gen_argss = data_gen_argss[:1]
    #     #     # epochss = epochss[:1]
shape = (64, 64)
if test == True:
    shape = (64, 64)

if remote == True:

    blacklist = utils.write_blacklist('/home/hendrik/src/mlebe/Blacklist')
if remote == False:
    blacklist = utils.write_blacklist('/Users/Hendrik/Documents/Semester_project/Blacklist')



pretrained = False
if remote == False:
    model_path1 = '/Users/Hendrik/Documents/mlebe_data/models/unet_ep300_val_loss0.10.hdf5'
else:
    model_path1 = '/Users/Hendrik/Desktop/new_hope0/training_results/dice_1200_2019-12-09/1_Step/unet_ep70_val_loss0.01.hdf5'


model_path2 = ''
model_path3 = ''
model_paths = [model_path1, '', ''] #needs to have the same lenght than epochss
pretrained_loss = ['dice', '', '']  #needs to have the same lenght than epochss
pretrained_step = 1
pretrained_seed = ''





for i, loss in enumerate(losses):
    if pretrained:
        if loss == pretrained_loss[i]:
            pretrained = True
        else: pretrained = False

    network_trainer.network_trainer(file_name, test, remote, loss, epochss, shape, data_gen_argss, min_epochs, max_tries, blacklist, remove_black_labels_and_columns = remove_black_labels_and_columns, visualisation=visualisation, pretrained = pretrained, pretrained_model_path= model_paths[i], pretrained_step = pretrained_step, pretrained_seed = pretrained_seed)


