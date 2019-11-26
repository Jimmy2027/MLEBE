import network_trainer

test = True
remote = False
visualisation = False  #if visualisation true saves pre- and unpreprocessed images for visualisation
# losses = ['dice_bincross', 'dice', 'bincross']
losses = ['dice']
epochs = 300
shape = (128, 128)
nmbr_tries = 1


for loss in losses:
    faulty_preds = True
    while faulty_preds == True:
        faulty_preds = network_trainer.network_trainer(test, remote, loss, epochs, shape, nmbr_tries)
        nmbr_tries += 1

