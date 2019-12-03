import network_trainer
import utils
import shutil
import tensorflow.keras

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


test = True
remote = False
visualisation = False  #if visualisation true saves pre- and unpreprocessed images for visualisation
# losses = ['dice', 'bincross', 'dice_bincross']
losses = ['bincross']
epochss = [300, 300, 300]
min_epochs = 60
data_gen_args3 = dict(rotation_range=90,
                     # brightness_range=[0.5, 1.2],
                     width_shift_range=30,
                     height_shift_range=30,
                     shear_range=5,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='nearest')

data_gen_args2 = dict(rotation_range=45,
                     # brightness_range=[0.5, 1.2],
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

data_gen_argss = [data_gen_args1, data_gen_args2, data_gen_args3]
max_tries = 3
shape = (128, 128)
if test == True:
    shape = (128, 128)

if remote == True:
    blacklist = utils.write_blacklist('/home/hendrik/src/mlebe/Blacklist')
if remote == False:
    blacklist = utils.write_blacklist('/Users/Hendrik/Documents/Semester_project/Blacklist')



pretrained = False
if pretrained:
    model_path1 = ''
    model_path2 = ''
    model_path3 = ''
    model_paths = [model_path1, model_path2, model_path3]
    step = ''


for i, loss in enumerate(losses):
    # if pretrained:
    #     # model = keras.models.load_model(model_paths[i], custom_objects={'dice_coef_loss': unet.dice_coef_loss})

    network_trainer.network_trainer(test, remote, loss, epochss, shape, data_gen_argss, min_epochs, max_tries, blacklist)


