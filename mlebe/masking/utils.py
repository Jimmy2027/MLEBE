import os

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
from jsonschema import Draft7Validator, validators
from scipy import ndimage

from mlebe.training.configs.utils import json_to_dict
from mlebe.training.dataio.transformation import get_dataset_transformation
from mlebe.training.utils.utils import json_file_to_pyobj
from mlebe.training.configs.utils import write_to_jsonfile


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
    Simply counts the number of unconnected objects in the volume and returns the second biggest one
    (the biggest is the black background)
    """
    markers = ndimage.label(image)[0]
    if len(np.unique(markers)) <= 2:
        return image
    l, counts = np.unique(markers, return_counts=True)
    brain_label = l[np.argsort(-counts)[1]]
    new = np.where(markers == brain_label, 1, 0)
    return new.astype('float64')


def get_masking_opts(workflow_config_path, input_type):
    # todo make schema for workflow config: needs to contain model path
    if input_type == 'anat':
        masking_opts = get_masking_anat_opts_defaults(
            json_to_dict(workflow_config_path)['masking_config'])['masking_config_anat']
    elif input_type == 'func':
        masking_opts = get_masking_func_opts_defaults(
            json_to_dict(workflow_config_path)['masking_config'])['masking_config_func']
    else:
        raise NotImplementedError(f'input type "{input_type}" is not implemented.')
    return masking_opts


def get_masking_anat_opts_defaults(config):
    """
    Fill the masking configuration file with defaults.

    Parameters
    ----------
    config : dict
            configuration of the func masking

    """
    DefaultValidatingDraft7Validator = extend_with_default(Draft7Validator)
    if 'masking_config_anat' not in config.keys():
        schema = {'properties': {
            'masking_config_anat': {'default': {
                "use_cuda": False,
                "with_bids_cropping": False,
                "input_type": "anat",
                "visualisation_bool": False,
                "bias_correct_bool": False,
                "test": False,
                "model_config_path": "",
                "crop_values": [15, 15]
            }}, }}
    else:
        schema = {'properties': {
            'masking_config_anat': {
                'properties': {
                    "use_cuda": {'default': False},
                    "with_bids_cropping": {'default': False},
                    "input_type": {'default': 'anat'},
                    "visualisation_bool": {'default': False},
                    "bias_correct_bool": {'default': False},
                    "test": {'default': False},
                    "model_folder_path": {'default': ''},
                    "crop_values": {'default': [15, 15]}
                }},
        }}
    DefaultValidatingDraft7Validator(schema).validate(config)
    return config


def get_masking_func_opts_defaults(config):
    """
    Fills the masking configuration file with defaults

    Parameters
    ----------
    config :  dict
        configuration of the func masking
    """
    DefaultValidatingDraft7Validator = extend_with_default(Draft7Validator)

    if 'masking_config_func' not in config.keys():
        schema = {'properties': {
            'masking_config_func': {'default': {
                "use_cuda": False,
                "with_bids_cropping": False,
                "input_type": "func",
                "visualisation_bool": False,
                "bias_correct_bool": False,
                "test": False,
                "model_config_path": "",
                "crop_values": [15, 15]
            }}, }}
    else:
        schema = {'properties': {
            'masking_config_func': {
                'properties': {
                    "use_cuda": {'default': False},
                    "with_bids_cropping": {'default': False},
                    "input_type": {'default': 'func'},
                    "visualisation_bool": {'default': False},
                    "bias_correct_bool": {'default': False},
                    "test": {'default': False},
                    "model_folder_path": {'default': ''},
                    "crop_values": {'default': [15, 15]}

                }},
        }}
    DefaultValidatingDraft7Validator(schema).validate(config)
    return config


def get_masking_opts_defaults(config):
    DefaultValidatingDraft7Validator = extend_with_default(Draft7Validator)

    schema = {'properties': {
        'workflow_config': {'default': {
            "with_FLASH": False,
            "subjects": [],
            "keep_work": True
        }}, }}
    DefaultValidatingDraft7Validator(schema).validate(config)
    return config


def get_model_config(masking_opts, return_path=False):
    """
    Returns model_config_path and writes model_path to it.
    """
    if masking_opts['test']:
        return {}
    model_folder_path = os.path.expanduser(masking_opts['model_folder_path'])
    for file in os.listdir(model_folder_path):
        if file.endswith('.json'):
            model_config_path = os.path.join(model_folder_path, file)
        if file.endswith('.pth'):
            model_path = os.path.join(model_folder_path, file)
    write_to_jsonfile(model_config_path, [('model.path_pre_trained_model', model_path)])
    if return_path:
        return model_config_path
    else:
        return json_file_to_pyobj(model_config_path)


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


def get_mask(json_opts, in_file_data, ori_shape):
    from mlebe.training.models import get_model
    # To make sure that the GPU is not used for the predictions: (might be unnecessary)
    if not json_opts.model.use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    model = get_model(json_opts.model)
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
    save_dir = os.path.join(workflow_config['visualisation_path'], os.path.basename(in_file))
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
        plt.savefig(save_dir + '/{}.{}'.format(slice, workflow_config['visualisation_format']),
                    format=workflow_config['visualisation_format'])
        plt.close()


def reconstruct_image(ori_shape, mask_pred):
    """
    The classifier was trained on images of different shape than the input image shape.
    That is why it needs to be resized before and after segmentation.
    Parameters
    ----------
    ori_shape : shape of the image that is to be masked
    mask_pred : predicted mask from the classifier

    Returns
    -------
    resized : predicted mask with the shape of the input image
    """
    resized = np.empty(ori_shape)
    for i, slice in enumerate(mask_pred):
        if ori_shape[1] < ori_shape[2]:
            padd = ori_shape[2] - ori_shape[1]
            resized_mask_temp = cv2.resize(slice, (ori_shape[2], ori_shape[2]))
            resized_mask = resized_mask_temp[padd // 2:ori_shape[1] + padd // 2, :]

        elif ori_shape[1] > ori_shape[2]:
            padd = ori_shape[1] - ori_shape[2]
            resized_mask_temp = cv2.resize(slice, (ori_shape[1], ori_shape[1]))
            resized_mask = resized_mask_temp[:, padd // 2:ori_shape[2] + padd // 2]
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


def extend_with_default(validator_class):
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for property, subschema in properties.items():
            if "default" in subschema:
                instance.setdefault(property, subschema["default"])

        for error in validate_properties(
                validator, properties, instance, schema,
        ):
            yield error

    return validators.extend(
        validator_class, {"properties": set_defaults},
    )
