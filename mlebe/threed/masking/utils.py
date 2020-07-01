import os
import cv2
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from mlebe.threed.training.dataio.transformation import get_dataset_transformation
from mlebe.threed.training.utils.utils import json_file_to_pyobj


def pred_volume_stats(mask_pred, save_path, file_name, model_path):
    unique, counts = np.unique(mask_pred, return_counts=True)
    volume = dict(zip(unique, counts))[1]
    if 'T2' in file_name:
        contrast = 'T2'
    elif 'bold' in file_name:
        contrast = 'BOLD'
    elif 'cbv' in file_name:
        contrast = 'CBV'
    if not os.path.isfile(os.path.join(save_path, 'pred_volume.csv')):
        pred_volume_df = pd.DataFrame(columns=['file_name', 'Contrast', 'Volume', 'model_path'])
    else:
        pred_volume_df = pd.read_csv(os.path.join(save_path, 'pred_volume.csv'))
    pred_volume_df = pred_volume_df.append(pd.DataFrame([[file_name, contrast, volume, model_path]],
                                                        columns=['file_name', 'Contrast', 'Volume', 'model_path']),
                                           sort=False)
    pred_volume_df.to_csv(os.path.join(save_path, 'pred_volume.csv'), index=False)
    return


def remove_outliers(image):
    """
    Simply counts the number of unconnected objects in the volume and returns the second biggest one (the biggest is the black background)
    """
    markers = ndimage.label(image)[0]
    if len(np.unique(markers)) > 2:
        l, counts = np.unique(markers, return_counts=True)
        brain_label = l[np.argsort(-counts)[1]]
        new = np.where(markers == brain_label, 1, 0)
        return new.astype('float64')
    else:
        return image


def get_workflow_config(workflow_config_path, input_type):
    if input_type == 'anat':
        workflow_config = json_file_to_pyobj(workflow_config_path).masking_config.masking_config_anat
    elif input_type == 'func':
        workflow_config = json_file_to_pyobj(workflow_config_path).masking_config.masking_config_func
    return workflow_config


def crop_bids_image(resampled_nii_path, crop_values=[20, 20]):
    """
    Cropping the bids image
    """
    resampled_bids_nib = nib.load(resampled_nii_path)
    resampled_bids = resampled_bids_nib.get_data()
    resampled_bids_cropped = resampled_bids[crop_values[0]:resampled_bids.shape[0] - crop_values[1], ...]
    resampled_bids_cropped_nib = nib.Nifti1Image(resampled_bids_cropped, resampled_bids_nib.affine,
                                                 resampled_bids_nib.header)
    nib.save(resampled_bids_cropped_nib, resampled_nii_path)


def get_mask(json_opts, in_file_data, model, ori_shape):
    ds_transform = get_dataset_transformation('mlebe', opts=json_opts.augmentation,
                                              max_output_channels=json_opts.model.output_nc)
    transformer = ds_transform['bids']()
    # preprocess data for compatibility with model
    network_input = transformer(np.expand_dims(in_file_data, -1))
    # add dimension for batches
    network_input = network_input.unsqueeze(0)
    model.set_input(network_input)
    model.test()
    # predict
    mask_pred = np.squeeze(model.pred_seg.cpu().byte().numpy()).astype(np.int16)
    # switching to z,x,y
    mask_pred = np.moveaxis(mask_pred, 2, 0)
    in_file_data = np.moveaxis(in_file_data, 2, 0)
    network_input = np.moveaxis(np.squeeze(network_input.cpu().numpy()), 2, 0)

    # need to un-pad on the z-axis to the original shape:
    diff = int(np.ceil(mask_pred.shape[0] - ori_shape[0]))
    mask_pred = mask_pred[int(np.ceil(diff / 2.)):  ori_shape[0] + int(np.ceil(diff / 2.)), :, :]
    network_input = network_input[int(np.ceil(diff / 2.)):  ori_shape[0] + int(np.ceil(diff / 2.)), :, :]

    return in_file_data, mask_pred, network_input


def save_visualisation(workflow_config, in_file, network_input, mask_pred):
    from matplotlib import pyplot as plt
    save_dir = os.path.join(workflow_config.visualisation_path, os.path.basename(in_file))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # pred_volume_stats(mask_pred, os.path.dirname(os.path.dirname(visualisation_path)), os.path.basename(in_file), model_path)
    for slice in range(network_input.shape[0]):
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(network_input[slice], cmap='gray')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(network_input[slice], cmap='gray')
        plt.imshow(mask_pred[slice], cmap='Blues', alpha=0.6)
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(mask_pred[slice])
        plt.axis('off')
        plt.savefig(save_dir + '/{}.{}'.format(slice, workflow_config.visualisation_format),
                    format=workflow_config.visualisation_format)
        plt.close()


def reconstruct_image(ori_shape, mask_pred):
    resized = np.empty(ori_shape)
    for i, slice in enumerate(mask_pred):
        if ori_shape[1] < ori_shape[2]:
            padd = ori_shape[2] - ori_shape[1]
            resized_mask_temp = cv2.resize(slice, (ori_shape[2], ori_shape[2]))
            resized_mask = resized_mask_temp[padd // 2:ori_shape[1] + padd // 2, :]

            resized[i] = resized_mask
        elif ori_shape[1] > ori_shape[2]:
            padd = ori_shape[1] - ori_shape[2]
            resized_mask_temp = cv2.resize(slice, (ori_shape[1], ori_shape[1]))
            resized_mask = resized_mask_temp[:, padd // 2:ori_shape[2] + padd // 2]
            resized[i] = resized_mask
        else:
            resized_mask = cv2.resize(slice, (ori_shape[2], ori_shape[1]))
            resized[i] = resized_mask

    # switching to x,y,z
    resized = np.moveaxis(resized, 0, 2)

    return resized


def pad_to_shape(resampled_mask_data, input_image_data):
    # it can happen that after forward and backward resampling the shape is not the same, this fixes that:
    if resampled_mask_data.shape < input_image_data.shape:
        resampled_mask_data = np.pad(resampled_mask_data, (
            (input_image_data.shape[0] - resampled_mask_data.shape[0], 0),
            (input_image_data.shape[1] - resampled_mask_data.shape[1], 0),
            (input_image_data.shape[2] - resampled_mask_data.shape[2], 0)), 'edge')
    else:
        resampled_mask_data = np.pad(resampled_mask_data, (
            (resampled_mask_data.shape[0] - input_image_data.shape[0], 0),
            (resampled_mask_data.shape[1] - input_image_data.shape[1], 0),
            (resampled_mask_data.shape[2] - input_image_data.shape[2], 0)), 'edge')

    return resampled_mask_data
