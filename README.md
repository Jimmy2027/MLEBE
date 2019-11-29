# MLEBE
Machine Learning Enabled Brain Extraction

### main.py

:param test: Bool: If Test is True, every parameter is set to increase learning speed. Used to test if the code runs

:param remote: Bool: If remote is True, the paths are set for remote computer

:param visualisation: Bool: if True, all images after preprocessing are saved

:param loss: Array of strings: with which loss the network will be trained

:param epochss: Array with epochs. Should have the same length than data_gen_argss

:param min_epochs: int: The minimum amount of epochs the network should be trained on. If this number is not reached, the training will start again with a different seed and reduced augmentation values

:param data_gen_argss: Array of dicts : arguments for the data augmentations, should have the same length than epochss

:param max_tries: int: Integer indicating how many times the training should be started again with reduced augmentation values

:param shape: Tuple (y,x): Shape of the images that should come out of the preprocessing