import os
import random

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataio.loaders import get_dataset
from dataio.transformation import get_dataset_transformation
from models import get_model
from utils import utils
from utils.utils import json_file_to_pyobj


def train(arguments):
    # Visualisation arguments
    with_mask = True
    only_mask = False
    len_x = 5  # number of images on x-axis for vis pdf
    len_y = 5  # number of images on y-axis for vis pdf
    nbr_pages = 20
    total_images = len_x * len_y * nbr_pages  # total number of images that will be augmented

    # Parse input arguments
    json_filename = arguments.config
    network_debug = arguments.debug

    # Load options
    json_opts = json_file_to_pyobj(json_filename)
    train_opts = json_opts.training

    # Architecture type
    arch_type = train_opts.arch_type

    # Setup Dataset and Augmentation
    ds_class = get_dataset('mlebe_dataset')
    ds_path = json_opts.data.data_dir
    template_path = json_opts.data.template_dir
    studies = json_opts.data.studies
    ds_transform = get_dataset_transformation('mlebe', opts=json_opts.augmentation,
                                              max_output_channels=json_opts.model.output_nc)

    # Setup channels
    channels = json_opts.data_opts.channels
    if len(channels) != json_opts.model.input_nc \
            or len(channels) != getattr(json_opts.augmentation, 'mlebe').scale_size[-1]:
        raise Exception(
            'Number of data channels must match number of model channels, and patch and scale size dimensions')

    # Setup Data Loader
    split_opts = json_opts.data_split
    train_dataset = ds_class(template_path, ds_path, studies, split='train', transform=ds_transform['train'],
                             train_size=split_opts.train_size, test_size=split_opts.test_size,
                             valid_size=split_opts.validation_size, split_seed=split_opts.seed)

    train_loader = DataLoader(dataset=train_dataset, num_workers=16, batch_size=train_opts.batchSize, shuffle=True)

    save_dir = os.path.join('visualisation', json_opts.model.experiment_name)
    utils.mkdirs(save_dir)

    slices = []
    masks = []

    for epoch_iter, (images, labels, indices) in tqdm(enumerate(train_loader, 1),
                                                      total=total_images):
        if epoch_iter <= total_images:

            images = images.numpy()
            labels = labels.numpy()

            for image_idx in range(images.shape[0]):
                image = np.squeeze(images[image_idx])
                label = np.squeeze(labels[image_idx])
                for slice in range(image.shape[0]):
                    if not np.max(image[slice]) < 0:
                        slices.append(image[slice])
                        masks.append(label[slice])

    temp = list(zip(slices, masks))

    random.shuffle(temp)

    slices, masks = zip(*temp)
    list_index = 1
    with PdfPages(save_dir + '/augm_vis.pdf') as pdf:
        for page in range(nbr_pages):
            plt.figure()
            idx = 1
            for slice in range(len_x * len_y):
                plt.subplot(len_y, len_x, idx)
                if only_mask:
                    plt.imshow(masks[list_index], cmap='gray')
                else:
                    plt.imshow(slices[list_index], cmap='gray')
                    if with_mask:
                        plt.imshow(masks[list_index], cmap='Blues', alpha=0.4)
                plt.axis('off')
                idx += 1
                list_index += 1
            pdf.savefig()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CNN Seg Training Function')

    parser.add_argument('-c', '--config', help='training config file', required=True)
    parser.add_argument('-d', '--debug', help='returns number of parameters and bp/fp runtime', action='store_true')
    args = parser.parse_args()

    train(args)
