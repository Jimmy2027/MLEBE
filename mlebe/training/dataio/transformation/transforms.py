from pprint import pprint

import numpy as np
import mlebe.training.dataio.torchsample_transforms as ts
# import torchvision.transforms as tv
from torchio.transforms.interpolation import Interpolation

from .imageTransformations import RandomElasticTransform, RandomAffineTransform, RandomNoiseTransform, \
    RandomFlipTransform, RandomBiasFieldTransform, get_normalization, Scale_mlebe


class Transformations:

    def __init__(self, name):
        self.name = name

        # Input patch and scale size
        self.scale_size = (192, 192, 1)
        self.patch_size = (128, 128, 1)

        # Affine and Intensity Transformations
        self.shift_val = (0, 0)
        self.rotate_val = 15.0
        self.inten_val = (1.0, 1.0)
        self.random_flip_prob = 0.0
        self.random_affine_prob = 0.0
        self.random_elastic_prob = 0.0
        self.random_noise_prob = 0.0
        self.bias_field_prob = 0.0
        self.scale_range = [1, 1]
        self.scale_proba = 0

        # Divisibility factor for testing
        self.division_factor = (16, 16, 1)

        # Maximum allowed for elastic transform
        self.max_deform = (7.5, 7.5, 7.5)

    def get_transformation(self):
        return {
            'gsd_pCT': self.get_gsd_pCT_transformer(),
            'mlebe': self.get_gsd_pCT_transformer()
        }[self.name]

    def print(self):
        print('\n\n############# Augmentation Parameters #############')
        pprint(vars(self))
        print('###################################################\n\n')

    def initialise(self, opts, max_output_channels=10):
        t_opts = getattr(opts, self.name)
        self.max_output_channels = max_output_channels

        # Affine and Intensity Transformations
        if hasattr(t_opts, 'scale_size'):       self.scale_size = t_opts.scale_size
        if hasattr(t_opts, 'scale_range'):        self.scale_range = t_opts.scale_range
        if hasattr(t_opts, 'scale_proba'):
            if self.scale_range == [1, 1]:
                self.scale_proba = 0
            else:
                self.scale_proba = t_opts.scale_proba
        if hasattr(t_opts, 'patch_size'):       self.patch_size = t_opts.patch_size
        if hasattr(t_opts, 'shift_val'):        self.shift_val = t_opts.shift_val
        if hasattr(t_opts, 'rotate'):           self.rotate_val = t_opts.rotate
        if hasattr(t_opts, 'max_deform'):       self.max_deform = t_opts.max_deform
        if hasattr(t_opts, 'inten_val'):        self.inten_val = t_opts.intensity
        if hasattr(t_opts, 'random_flip_prob'): self.random_flip_prob = t_opts.random_flip_prob
        if hasattr(t_opts, 'random_affine_prob'): self.random_affine_prob = t_opts.random_affine_prob
        if hasattr(t_opts, 'random_elastic_prob'): self.random_elastic_prob = t_opts.random_elastic_prob
        if hasattr(t_opts, 'bias_field_prob'):  self.bias_field_prob = t_opts.bias_field_prob
        if hasattr(t_opts, 'bias_magnitude_range'):  self.bias_magnitude_range = t_opts.bias_magnitude_range
        if hasattr(t_opts, 'division_factor'):  self.division_factor = t_opts.division_factor
        if hasattr(t_opts, 'random_noise_prob'):  self.random_noise_prob = t_opts.random_noise_prob
        if hasattr(t_opts, 'normalization'):  self.normalization = t_opts.normalization

    def get_gsd_pCT_transformer(self):
        return {'train': self.gsd_pCT_train_transform, 'valid': self.gsd_pCT_valid_transform,
                'bids': self.bids_transform}

    def gsd_pCT_train_transform(self, seed=None):
        if seed is None:
            # seed must be an integer for torch
            seed = np.random.randint(0, 9999)

        train_transform = ts.Compose([
            Scale_mlebe(self.scale_range, self.scale_size, self.scale_proba),
            ts.ToTensor(),
            ts.Pad(size=self.scale_size),
            ts.TypeCast(['float', 'float']),

            RandomFlipTransform(axes=(1), p=self.random_flip_prob, seed=seed,
                                max_output_channels=self.max_output_channels),
            RandomElasticTransform(seed=seed, p=self.random_elastic_prob, image_interpolation=Interpolation.BSPLINE,
                                   max_displacement=self.max_deform,
                                   max_output_channels=self.max_output_channels),

            RandomAffineTransform(scales=[1, 1], degrees=(self.rotate_val), isotropic=True, default_pad_value=0,
                                  image_interpolation=Interpolation.BSPLINE, seed=seed, p=self.random_affine_prob,
                                  max_output_channels=self.max_output_channels),

            RandomNoiseTransform(p=self.random_noise_prob, seed=seed, max_output_channels=self.max_output_channels),

            # somehow applies shift even if "shift": [0,0]
            ts.RandomAffine(translation_range=self.shift_val),

            RandomBiasFieldTransform(p=self.bias_field_prob, is_tensor=True,
                                     bias_magnitude_range=self.bias_magnitude_range),

            ts.ChannelsFirst(),
            # ts.NormalizeMedicPercentile(norm_flag=(True, False)),
            get_normalization(self.normalization),
            # Normalize_mlebe(),
            # ts.NormalizeMedic(norm_flag=(True, False)),
            # Todo fork torchsample and fix the Random Crop bug
            # ts.ChannelsLast(), # seems to be needed for crop
            # ts.RandomCrop(size=self.patch_size),
            ts.TypeCast(['float', 'long'])
        ])

        return train_transform

    def gsd_pCT_valid_transform(self, seed=None):
        valid_transform = ts.Compose([
            Scale_mlebe([0, 0], self.scale_size, 0),
            ts.ToTensor(),
            ts.Pad(size=self.scale_size),
            ts.ChannelsFirst(),
            ts.TypeCast(['float', 'float']),
            # ts.NormalizeMedicPercentile(norm_flag=(True, False)),
            # ts.NormalizeMedic(norm_flag=(True, False)),
            get_normalization(self.normalization),
            # ts.ChannelsLast(),
            # ts.SpecialCrop(size=self.patch_size, crop_type=0),
            ts.TypeCast(['float', 'long'])
        ])
        return valid_transform

    def bids_transform(self):
        bids_transform = ts.Compose([
            Scale_mlebe([0, 0], self.scale_size, 0, bids=True),
            ts.ToTensor(),
            ts.Pad(size=self.scale_size),
            ts.ChannelsFirst(),
            ts.TypeCast(['float', 'float']),
            # ts.NormalizeMedicPercentile(norm_flag=(True, False)),
            # ts.NormalizeMedic(norm_flag=(True, False)),
            get_normalization(self.normalization),
            # ts.ChannelsLast(),
            # ts.SpecialCrop(size=self.patch_size, crop_type=0),
            ts.TypeCast(['float', 'long'])
        ])

        return bids_transform
