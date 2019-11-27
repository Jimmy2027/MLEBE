import network_trainer

test = True
remote = False
visualisation = False  #if visualisation true saves pre- and unpreprocessed images for visualisation
losses = ['dice', 'dice_bincross', 'bincross']
# losses = ['bincross', 'dice']
epochs = 20
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
        faulty_preds = network_trainer.network_trainer(test, remote, loss, epochs, shape, nmbr_tries)


