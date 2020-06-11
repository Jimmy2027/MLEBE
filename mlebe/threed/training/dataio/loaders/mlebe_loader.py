import datetime
import os
import uuid
import nibabel as nib
import numpy as np
import pandas as pd
from mlebe.training.utils import data_loader as dl
from mlebe.training.utils.general import data_normalization, arrange_mask, write_blacklist
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
from mlebe.threed.training.dataio.loaders.utils import validate_images


class mlebe_dataset(Dataset):

    def __init__(self, template_dir, data_dir, data_opts, split, save_dir=None, transform=None,
                 split_seed=42, train_size=0.7, test_size=0.15, valid_size=0.15):
        """
        if train_size = None, no splitting of the data is done
        """
        super(mlebe_dataset, self).__init__()
        self.save_dir = save_dir
        self.transform = transform
        self.template_dir = template_dir
        self.split = split
        self.data_type = data_opts.data_type
        self.with_arranged_maks = data_opts.with_arranged_maks
        self.with_blacklist = data_opts.with_blacklist
        if self.with_blacklist:
            self.blacklist = write_blacklist(os.path.expanduser(data_opts.blacklist_dir))
        if data_opts.data_type == 'anat':
            self.data_selection = self.make_dataselection_anat(data_dir, data_opts.studies)
        elif data_opts.data_type == 'func':
            self.data_selection = self.make_dataselection_func(data_dir, data_opts.studies)
        else:
            assert False, 'Wrong data_type defined for {} dataset: {}, choose between anat and func'.format(split,
                                                                                                            data_opts.data_type)

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

    def make_dataselection_anat(self, data_dir, studies):
        data_selection = pd.DataFrame()
        for o in os.listdir(data_dir):
            if (not studies or o in studies) and not o.startswith('.') and not o.endswith(
                    '.xz'):  # i.e. if o in studies or if studies empty
                print(o)
                data_set = o
                for x in os.listdir(os.path.join(data_dir, o)):
                    if x.endswith('preprocessing') or x.startswith('preprocess') and not x.endswith('work'):
                        for root, dirs, files in os.walk(os.path.join(data_dir, o, x)):
                            for file in files:
                                if not file.startswith('.') and (
                                        file.endswith("_T2w.nii.gz") or file.endswith("_T1w.nii.gz")):
                                    split = file.split('_')
                                    subject = split[0].split('-')[1]
                                    session = split[1].split('-')[1]
                                    acquisition = split[2].split('-')[1]
                                    type = split[3].split('.')[0]
                                    uid = file.split('.')[0]
                                    path = os.path.join(root, file)
                                    blacklisted = False
                                    if self.with_blacklist:
                                        for i in self.blacklist:
                                            if subject == i.subj and session == i.sess:
                                                blacklisted = True
                                    if blacklisted == False:
                                        data_selection = pd.concat([data_selection, pd.DataFrame(
                                            [[data_set, subject, session, acquisition, type, uid, path]],
                                            columns=['data_set', 'subject', 'session', 'acquisition', 'type', 'uid',
                                                     'path'])])
        if self.save_dir:
            data_selection.to_csv(os.path.join(self.save_dir, self.split + '_dataset.csv'), index=False)
        return data_selection

    def make_dataselection_func(self, data_dir, studies):
        data_selection = pd.DataFrame()

        func_training_dir = os.path.abspath(os.path.expanduser('~/var/tmp/func_training'))

        if not os.path.exists(func_training_dir):
            print('creating dir: ', func_training_dir)
            os.makedirs(func_training_dir)
        for o in os.listdir(data_dir):
            if o in studies and not o.startswith('.') and not o.startswith('.') and not o.endswith('.xz'):
                data_set = o
                for x in os.listdir(os.path.join(data_dir, o)):
                    if x.endswith('preprocessing') or x.startswith('preprocess') and not x.endswith('work'):
                        for root, dirs, files in os.walk(os.path.join(data_dir, o, x)):
                            if root.endswith('func'):
                                for file in files:
                                    if file.endswith(".nii.gz"):
                                        tMean_path = os.path.join(func_training_dir, 'tMean_' + file)
                                        # collapse volumes over time
                                        if not os.path.isfile(tMean_path):
                                            command = 'fslmaths {a} -Tmean {b}'.format(a=os.path.join(root, file),
                                                                                       b=tMean_path)
                                            print(command)
                                            os.system(command)

                                        split = file.split('_')
                                        subject = split[0].split('-')[1]
                                        session = split[1].split('-')[1]
                                        acquisition = split[2].split('-')[1]
                                        type = split[3].split('.')[0]
                                        uid = file.split('.')[0]
                                        path = tMean_path
                                        data_selection = pd.concat([data_selection, pd.DataFrame(
                                            [[data_set, subject, session, acquisition, type, uid, path]],
                                            columns=['data_set', 'subject', 'session', 'acquisition', 'type', 'uid',
                                                     'path'])])
        if self.save_dir:
            data_selection.to_csv(os.path.join(self.save_dir, self.split + '_dataset.csv'), index=False)
        return data_selection

    def get_ids(self, indices):
        return [self.ids[index] for index in indices]

    def __len__(self):
        return len(self.selection)

    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        img = nib.load(self.selection.iloc[index]['path']).get_data()
        target = dl.load_mask(self.template_dir).get_data()

        if self.with_arranged_maks:
            # set the mask to zero where the image is zero
            target = arrange_mask(img, target)

        # img = preprocess(img, (128, 128), 'coronal')
        # target = preprocess(target, (128, 128), 'coronal')
        img = np.moveaxis(img, 2, 1)
        target = np.moveaxis(target, 2, 1)

        # Make sure there is a channel dimension
        img = np.expand_dims(img, axis=-1)
        target = np.expand_dims(target, axis=-1)

        # handle exceptions
        validate_images(img, target)

        # apply transformations
        if self.transform:
            transformer = self.transform()
            img, target = transformer(img, target)
        img = torch.from_numpy(data_normalization(img.numpy()))
        return img, target, index


class experiment_config():
    def __init__(self, json_config, pretrained_model=False):
        self.json_config = json_config
        self.pretrained_model = pretrained_model
        self.experiment_config = self.make_experiment_config_df()

    def make_experiment_config_df(self):
        experiment_config = pd.DataFrame([[]])
        # experiment_config['data_sets'] = str(self.json_config.data.studies)
        # experiment_config['excluded'] = self.json_config.data.excluded_from_training
        # experiment_config['slice_view'] = self.json_config.data.slice_view
        experiment_config['pretrained_model'] = self.pretrained_model
        # experiment_config['data_type'] = self.json_config.data.data_type
        experiment_config['uid'] = uuid.uuid4()
        experiment_config['loss'] = self.json_config.model.criterion
        experiment_config['blacklist'] = False
        experiment_config['model'] = self.json_config.model.model_type
        experiment_config['lr'] = self.json_config.model.lr_rate
        experiment_config['date_time'] = str(datetime.datetime.now())
        experiment_config['augmentation_params'] = str(self.json_config.augmentation.mlebe)
        experiment_config['shape'] = str(self.json_config.augmentation.mlebe.scale_size)

        data_config = self.json_config.data._asdict()
        for key, value in zip(data_config.keys(), data_config.values()):
            if not (key == 'data_dir' or key == 'template_dir'):
                if key == "studies":
                    experiment_config[key] = str(value)
                else:
                    experiment_config[key] = value

        return experiment_config

    def save(self, experiment_config_name='results'):
        if not os.path.exists(experiment_config_name + '.csv'):
            self.experiment_config.to_csv(experiment_config_name + '.csv', index=False)
        else:
            old_experiment_results = pd.read_csv(experiment_config_name + '.csv')
            new_experiment_results = pd.concat([old_experiment_results, self.experiment_config])
            new_experiment_results.to_csv(experiment_config_name + '.csv', index=False)
