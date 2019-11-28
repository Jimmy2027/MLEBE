import network_trainer

test = True
remote = False
visualisation = False  #if visualisation true saves pre- and unpreprocessed images for visualisation
# losses = ['dice', 'dice_bincross', 'bincross']
losses = ['bincross']
epochs = [100, 100, 100]
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
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest')

data_gen_argss = [data_gen_args1, data_gen_args2]
max_tries = 5
shape = (128, 128)
if test == True:
    shape = (32, 32)



for loss in losses:
    faulty_preds = True
    nmbr_tries = 0
    while (faulty_preds == True) and (nmbr_tries < max_tries + 1):
        nmbr_tries += 1
        print('Number of tries: ', nmbr_tries)
        faulty_preds = network_trainer.network_trainer(test, remote, loss, epochs, shape, nmbr_tries, data_gen_argss, min_epochs)


