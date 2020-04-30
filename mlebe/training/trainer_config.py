import os
import mlebe.training.utils.general
from mlebe.training.network_trainer import network_trainer
import tensorflow as tf


class trainer_config:

    def __init__(self, file_name, data_dir, template_dir, slice_view, shape, loss, epochss, data_gen_argss,
                 blacklist=False, visualisation=False, test=False, pretrained_model=False, data_type='anat',
                 excluded_from_training=[''], data_sets=[], blacklist_dir='', batch_size=64, model_comparison_df='',
                 tensorboard_bool=False, model = 'unet'):
        """
        :param test: Bool: If Test is True, every parameter is set to increase learning speed. Used to test if the code runs
        :param visualisation: Bool: if True, saves pre- and unpreprocessed images for visualisation (SLOW!)
        :param loss: loss with which the network will be trained
        :param epochss: Array with epochs. Should have the same length than data_gen_argss
        :param min_epochs: int: The minimum amount of epochs the network should be trained on. If this number is not reached, the training will start again with reduced augmentation values
        :param data_gen_argss: Array of dicts : arguments for the data augmentations, should have the same length than epochss, None for no augmentations
        :param max_tries: int: Integer indicating how many times the training should be started again with reduced augmentation values
        :param shape: Tuple (y,x): Shape of the images that should come out of the preprocessing
        :param model_comparison_df: path to Dataframe where the results can be stored and compared
        """

        self.file_name = file_name
        self.pretrained_model = pretrained_model
        self.data_dir = data_dir
        self.template_dir = template_dir
        self.blacklist = blacklist
        self.slice_view = slice_view
        self.data_type = data_type
        self.shape = shape
        self.visualisation = visualisation
        self.test = test
        self.loss = loss
        self.data_sets = data_sets
        self.excluded_from_training = excluded_from_training
        self.epochss = epochss
        self.data_gen_argss = data_gen_argss
        self.trained_model_path = ''
        self.blacklist_dir = blacklist_dir
        self.batch_size = batch_size
        self.model_comparison_df = model_comparison_df
        self.tensorboard_bool = tensorboard_bool
        self.model = model

    def train(self):

        if self.test == True:
            min_epochs = 0
        if self.data_type == 'func':
            blacklist = False

        if self.blacklist == True:
            if os.path.isdir(self.blacklist_dir):
                self.blacklist = mlebe.training.utils.general.write_blacklist(self.blacklist_dir)
            else:
                self.blacklist = False
                print('No Blacklist dir found')

        network_trainer(self, self.file_name, self.data_dir, self.template_dir, self.test, self.loss,
                        self.epochss, self.shape, self.data_gen_argss, self.blacklist, self.data_type,
                        slice_view=self.slice_view, visualisation=self.visualisation,
                        pretrained_model=self.pretrained_model, data_sets=self.data_sets,
                        excluded_from_training=self.excluded_from_training)
        tf.keras.backend.clear_session()

    def set_trained_model_path(self, path):
        self.trained_model_path = path
