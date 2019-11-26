import network_trainer

test = False
remote = True
visualisation = False  #if visualisation true saves pre- and unpreprocessed images for visualisation
# losses = ['dice_bincross', 'dice', 'bincross']
losses = ['bincross', 'dice']
epochs = 20
shape = (128, 128)
nmbr_tries = 0


for loss in losses:
    faulty_preds = True
    while faulty_preds == True:
        nmbr_tries += 1
        print('Number of tries: ', nmbr_tries)
        faulty_preds = network_trainer.network_trainer(test, remote, loss, epochs, shape, nmbr_tries)


