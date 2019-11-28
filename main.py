import network_trainer

test = True
remote = False
visualisation = False  #if visualisation true saves pre- and unpreprocessed images for visualisation
losses = ['dice', 'bincross', 'dice_bincross']
# losses = ['bincross']
epochss = [100, 100, 100]
min_epochs = 15
data_gen_args3 = dict(rotation_range=90,
                     brightness_range=[0.5, 1.2],
                     width_shift_range=30,
                     height_shift_range=30,
                     shear_range=5,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='nearest')

data_gen_args2 = dict(rotation_range=45,
                     brightness_range=[0.5, 1.2],
                     width_shift_range=15,
                     height_shift_range=15,
                     shear_range=5,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='nearest')

data_gen_args1 = dict(rotation_range=0.2,
                    brightness_range=[0.8, 1.1], #if training step 1 doesn0t work anymore it's because I added this line
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest')

data_gen_argss = [data_gen_args1, data_gen_args2, data_gen_args3]
max_tries = 5
shape = (128, 128)
if test == True:
    shape = (32, 32)

# if pretrained:



for loss in losses:

    network_trainer.network_trainer(test, remote, loss, epochss, shape, data_gen_argss, min_epochs, max_tries)


