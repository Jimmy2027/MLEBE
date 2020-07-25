import datetime
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from mlebe.training.dataio.loaders.utils import arrange_mask, write_blacklist
from mlebe.training.dataio.loaders.utils import validate_images


class test_dataset(Dataset):

    def __init__(self, template_dir, data_dir, data_opts, split, save_dir=None, transform=None,
                 split_seed=42, train_size=0.7, test_size=0.15, valid_size=0.15, training_shape=(128, 128, 96)):
        """
        if train_size = None, no splitting of the data is done
        """
        super(test_dataset, self).__init__()
        self.save_dir = save_dir
        self.data_opts = data_opts
        self.transform = transform
        self.template_dir = template_dir
        self.split = split
        self.data_type = data_opts.data_type
        self.with_arranged_mask = data_opts.with_arranged_mask
        self.training_shape = training_shape
        if 'with_blacklist' in data_opts._fields:
            self.with_blacklist = data_opts.with_blacklist
        else:
            self.with_blacklist = False
        if self.with_blacklist:
            self.blacklist = write_blacklist(os.path.expanduser(data_opts.blacklist_dir))
        if train_size:
            test_valid_size = test_size + valid_size

            train_selection, test_val_selection = train_test_split(self.data_selection, train_size=train_size,
                                                                   test_size=test_valid_size,
                                                                   random_state=split_seed)
            test_selection, validation_selection = train_test_split(test_val_selection,
                                                                    train_size=test_size / test_valid_size,
                                                                    test_size=valid_size / test_valid_size,
                                                                    random_state=split_seed)

            if split == 'train':
                self.selection = train_selection
            if split == 'test':
                self.selection = test_selection
                if data_opts.data_type == 'anat':
                    excluded_dataselection = self.make_dataselection_anat(data_dir, data_opts.excluded_from_training)
                elif data_opts.data_type == 'func':
                    excluded_dataselection = self.make_dataselection_func(data_dir, data_opts.excluded_from_training)

                self.selection = pd.concat([self.selection, excluded_dataselection])
            if split == 'validation':
                self.selection = validation_selection

        else:
            self.selection = self.data_selection

        self.ids = self.selection['uid'].to_list()

    def get_ids(self, indices):
        return [self.ids[index] for index in indices]

    def __len__(self):
        return len(1)

    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        img = np.empty((63, 96, 48))
        target = np.empty((63, 96, 48))

        if self.with_arranged_mask:
            # set the mask to zero where the image is zero
            target = arrange_mask(img, target)

        # Make sure there is a channel dimension
        img = np.expand_dims(img, axis=-1)
        target = np.expand_dims(target, axis=-1)

        # handle exceptions
        validate_images(img, target)

        # apply transformations
        if self.transform:
            transformer = self.transform()
            img, target = transformer(img, target)
        return img, target, index

    def preprocess_volume(self, volume):
        if self.data_opts.data_dimension_format == 'x,y,z':
            volume = np.moveaxis(volume, 2, 1)
        # Make sure there is a channel dimension
        volume = np.expand_dims(volume, axis=-1)

        # apply transformations
        if self.transform:
            transformer = self.transform()
            volume = transformer(volume)
        return volume
