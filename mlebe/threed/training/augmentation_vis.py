import os
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import DataLoader
from tqdm import tqdm
import nibabel as nib
from mlebe.threed.training.dataio.loaders import get_dataset
from mlebe.threed.training.dataio.transformation import get_dataset_transformation
from mlebe.threed.training.models import get_model
from mlebe.threed.training.utils import utils
from mlebe.threed.training.utils.utils import json_file_to_pyobj
import torchio

print(torchio.__file__)
"""
Notes:
shift is working
rotation is working
noise is working

rot + shift takes wrong interpolation for mask
skewing introduces weird artifacts, same with scaling
"""


# todo images need to be in form X,Y,Z
def aug_vis(json_filename):
    # Visualisation arguments
    with_mask = True
    shuffle_slices = True
    len_x = 5  # number of images on x-axis for vis pdf
    len_y = 5  # number of images on y-axis for vis pdf
    nbr_pages = 20
    total_images = len_x * len_y * nbr_pages  # total number of slices that will be augmented

    # Load options
    json_opts = json_file_to_pyobj(json_filename)
    train_opts = json_opts.training

    # Architecture type
    arch_type = train_opts.arch_type

    # Setup Dataset and Augmentation
    ds_class = get_dataset('mlebe_dataset')
    ds_path = json_opts.data.data_dir
    template_path = json_opts.data.template_dir
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
    train_dataset = ds_class(template_path, ds_path, json_opts.data, split='train',
                             transform=ds_transform['train'],
                             train_size=split_opts.train_size, test_size=split_opts.test_size,
                             valid_size=split_opts.validation_size, split_seed=split_opts.seed,
                             training_shape=json_opts.augmentation.mlebe.scale_size[:3])
    valid_dataset = ds_class(template_path, ds_path, json_opts.data, split='train',
                             transform=ds_transform['valid'],
                             train_size=split_opts.train_size, test_size=split_opts.test_size,
                             valid_size=split_opts.validation_size, split_seed=split_opts.seed,
                             training_shape=json_opts.augmentation.mlebe.scale_size[:3])

    train_dataset.selection = train_dataset.selection[:(total_images // 96) * 3]
    valid_dataset.selection = valid_dataset.selection[:(total_images // 96) * 3]

    train_loader = DataLoader(dataset=train_dataset, num_workers=16, batch_size=train_opts.batchSize, shuffle=False)
    valid_loader = DataLoader(dataset=valid_dataset, num_workers=16, batch_size=train_opts.batchSize, shuffle=False)

    save_dir = os.path.join('visualisation', json_opts.model.experiment_name)
    utils.rm_and_mkdir(save_dir)

    slices = []
    masks = []
    slice_ids = []

    val_slices = []
    val_masks = []
    val_slice_ids = []
    # original slices
    ori_slices = []

    data_selection = train_dataset.selection

    # for epoch_iter, [(images, labels, indices), (val_images, val_labels, val_indices)] in enumerate(
    #         tqdm(zip(train_loader, valid_loader))):
    #     print(len(images))
    #     asdfds

    for epoch_iter, [(images, labels, indices), (val_images, val_labels, val_indices)] in tqdm(
            enumerate(zip(train_loader, valid_loader), 1),
            total=len(train_loader)):
        if epoch_iter <= total_images:

            images = images.numpy()
            labels = labels.numpy()
            ids = train_dataset.get_ids(indices)

            val_images = val_images.numpy()
            val_labels = val_labels.numpy()
            val_ids = valid_dataset.get_ids(val_indices)
            assert ids == val_ids

            for image_idx in range(images.shape[0]):
                original_image_path = data_selection.loc[data_selection['uid'] == ids[image_idx], 'path'].item()
                orignal_image = nib.load(original_image_path).get_data()
                orignal_image = np.moveaxis(orignal_image, 2, 1)

                image = np.squeeze(images[image_idx])
                label = np.squeeze(labels[image_idx])

                val_image = np.squeeze(val_images[image_idx])
                val_label = np.squeeze(val_labels[image_idx])
                assert image.shape == val_image.shape, 'augm shape {} does not correspond to val image shape {}'.format(
                    image.shape, val_image.shape)

                for slice in range(image.shape[2]):
                    if not np.max(image[..., slice]) <= 0:
                        slices.append(image[..., slice])
                        masks.append(label[..., slice])
                        slice_ids.append(ids[image_idx] + '_slice{}'.format(slice))

                        val_slices.append(val_image[..., slice])
                        val_masks.append(val_label[..., slice])
                        val_slice_ids.append(val_ids[image_idx] + '_slice{}'.format(slice))

                        ori_slices.append(orignal_image[..., slice])

    if shuffle_slices:
        random.Random(42).shuffle(slices)
        random.Random(42).shuffle(masks)
        random.Random(42).shuffle(slice_ids)

    list_index = 1
    with PdfPages(save_dir + '/augm_img_vis.pdf') as pdf:
        for page in range(nbr_pages):
            plt.figure()
            plt.figtext(.05, 0.95, str(json_opts.augmentation.mlebe), fontsize=4)
            idx = 1
            for slice in range(len_x * len_y):
                plt.subplot(len_y, len_x, idx)
                plt.imshow(slices[list_index], cmap='gray')
                if with_mask:
                    plt.imshow(masks[list_index], cmap='Blues', alpha=0.4)
                plt.axis('off')
                plt.title(slice_ids[list_index], fontsize=2)
                idx += 1
                list_index += 1
            pdf.savefig()
            plt.close()

    list_index = 1
    with PdfPages(save_dir + '/augm_mask_vis.pdf') as pdf:
        for page in range(nbr_pages):
            plt.figure()
            plt.figtext(.05, 0.95, str(json_opts.augmentation.mlebe), fontsize=4)
            idx = 1
            for slice in range(len_x * len_y):
                plt.subplot(len_y, len_x, idx)
                plt.title(slice_ids[list_index], fontsize=2)
                plt.imshow(masks[list_index], cmap='gray')
                plt.axis('off')
                idx += 1
                list_index += 1
            pdf.savefig()
            plt.close()

    list_index = 1
    with PdfPages(save_dir + '/val_augm_img_vis.pdf') as pdf:
        for page in range(nbr_pages):
            plt.figure()
            plt.figtext(.05, 0.95, 'Valid augm', fontsize=4)
            idx = 1
            for slice in range(len_x * len_y):
                plt.subplot(len_y, len_x, idx)
                plt.imshow(val_slices[list_index], cmap='gray')
                if with_mask:
                    plt.imshow(val_masks[list_index], cmap='Blues', alpha=0.4)
                plt.axis('off')
                plt.title(val_slice_ids[list_index], fontsize=2)
                idx += 1
                list_index += 1
            pdf.savefig()
            plt.close()

    list_index = 1
    with PdfPages(save_dir + '/ori_img_vis.pdf') as pdf:
        for page in range(nbr_pages):
            plt.figure()
            plt.figtext(.05, 0.95, 'Valid augm', fontsize=4)
            idx = 1
            for slice in range(len_x * len_y):
                plt.subplot(len_y, len_x, idx)
                plt.imshow(ori_slices[list_index], cmap='gray')
                # if with_mask:
                #     plt.imshow(val_masks[list_index], cmap='Blues', alpha=0.4)
                plt.axis('off')
                plt.title(val_slice_ids[list_index], fontsize=2)
                idx += 1
                list_index += 1
            pdf.savefig()
            plt.close()


# config_paths = ['/home/hendrik/src/MLEBE/mlebe/threed/training/configs/augm_tries/elastic.json',
#                 '/home/hendrik/src/MLEBE/mlebe/threed/training/configs/augm_tries/rotation.json',
#                 '/home/hendrik/src/MLEBE/mlebe/threed/training/configs/augm_tries/shift.json',
#                 '/home/hendrik/src/MLEBE/mlebe/threed/training/configs/augm_tries/scale.json',
#                 '/home/hendrik/src/MLEBE/mlebe/threed/training/configs/augm_tries/flip.json',
#                 '/home/hendrik/src/MLEBE/mlebe/threed/training/configs/augm_tries/noise.json',
#                 '/home/hendrik/src/MLEBE/mlebe/threed/training/configs/try_augm.json']

config_paths = [
    '/home/hendrik/src/MLEBE/mlebe/threed/training/configs/mlebe_config_anat.json',
]
for config_path in config_paths:
    aug_vis(config_path)
