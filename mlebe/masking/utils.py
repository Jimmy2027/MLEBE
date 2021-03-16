import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import nibabel as nib
import numpy as np
from jsonschema import Draft7Validator, validators
from scipy import ndimage

from mlebe import log
from mlebe.training.configs.utils import json_to_dict, write_to_jsonfile
from mlebe.training.dataio.transformation import get_dataset_transformation
from mlebe.training.utils.utils import json_file_to_pyobj

DEFAULT_CONFIG_PATH = Path(__file__).parent / 'config/default_schema.json'


def get_mlebe_models(input_type: str) -> Path:
    """
    Get the path to the pretrained mlebe classifiers. If they don't exist under data/ they are downloaded there.
    Parameters
    ----------
    input_type : str
        either 'func' for CDV or BOLD contrast or 'anat' for T2 contrast

    Returns
    -------
    Path to the model folder.
    """
    download_urls = {'anat': 'https://zenodo.org/record/4031286/files/3D_unet_EPI.zip',
                     'func': 'https://zenodo.org/record/4031286/files/3D_unet_RARE.zip'}
    data_path = Path(__file__).parent / 'data'
    if not data_path.exists():
        data_path.mkdir()
    model_folder_paths = {'anat': data_path / '3D_unet_EPI', 'func': data_path / '3D_unet_RARE'}
    if not model_folder_paths[input_type].exists():
        with tempfile.TemporaryDirectory() as tmpdirname:
            wget_command = f'wget {download_urls[input_type]} -P {tmpdirname}/'
            log.info(f'Getting {input_type} model with "{wget_command}"')
            os.system(wget_command)

            unzip_command = f'unzip {tmpdirname}/{model_folder_paths[input_type].stem + ".zip"} -d {data_path}/'
            os.system(unzip_command)

    return model_folder_paths[input_type]


def remove_outliers(image):
    """
    Count the number of unconnected objects in the volume and return the second biggest one
    (the biggest being the black background).
    """
    markers = ndimage.label(image)[0]
    if len(np.unique(markers)) <= 2:
        return image
    l, counts = np.unique(markers, return_counts=True)
    brain_label = l[np.argsort(-counts)[1]]
    new = np.where(markers == brain_label, 1, 0)
    return new.astype('float64')


def get_masking_opts(masking_config_path: Optional[str], input_type: str):
    """Read the json config from the masking_config_path and fill the defaults with a schema."""
    config = json_to_dict(masking_config_path)['masking_config'] if masking_config_path else {}

    return get_masking_opts_defaults(config, input_type)[f'masking_config_{input_type}']


def get_biascorrect_opts_defaults(config: dict):
    """
    Fill the masking configuration file with defaults.

    Parameters
    ----------
    config : dict
            configuration of the func masking

    >>> get_biascorrect_opts_defaults({})
    {'bias_field_correction': {'bspline_fitting': '[10, 4]', 'convergence': '[ 150x100x50x30, 1e-16 ]', 'shrink_factor': 2}}
    >>> get_biascorrect_opts_defaults({'bias_field_correction': {'bspline_fitting': '[2, 400]'}})
    {'bias_field_correction': {'bspline_fitting': '[2, 400]', 'convergence': '[ 150x100x50x30, 1e-16 ]', 'shrink_factor': 2}}
    """
    DefaultValidatingDraft7Validator = extend_with_default(Draft7Validator)
    if 'bias_field_correction' not in config.keys():
        config['bias_field_correction'] = {}

    with open(DEFAULT_CONFIG_PATH, 'r') as json_file:
        schema = json.load(json_file)

    schema = {'properties': {'bias_field_correction': schema['bias_field_correction']}}
    DefaultValidatingDraft7Validator(schema).validate(config)
    return config['bias_field_correction']


def get_masking_opts_defaults(config: dict, input_type: str):
    """
    Fill the masking configuration file with defaults.

    Parameters
    ----------
    config : dict
            masking configuration dict
    input_type : str
        either 'func' for CDV or BOLD contrast or 'anat' for T2 contrast
    """
    DefaultValidatingDraft7Validator = extend_with_default(Draft7Validator)
    if f'masking_config_{input_type}' not in config.keys():
        config[f'masking_config_{input_type}'] = {}

    with open(DEFAULT_CONFIG_PATH, 'r') as json_file:
        schema = json.load(json_file)

    schema = {'properties': {f'masking_config_{input_type}': schema[f'masking_config_{input_type}']}}
    DefaultValidatingDraft7Validator(schema).validate(config)
    return config


def get_model_config(masking_opts, return_path=False):
    """
    Returns model_config_path and writes model_path to it.
    """
    model_folder_path = os.path.expanduser(masking_opts['model_folder_path'])
    for file in os.listdir(model_folder_path):
        if file.endswith('.json'):
            model_config_path = os.path.join(model_folder_path, file)
        if file.endswith('.pth'):
            model_path = os.path.join(model_folder_path, file)
    assert model_config_path, f'Model config path was not found under "{model_folder_path}"'
    assert model_path, f'Model path was not found under "{model_path}"'

    log.info(f'Writing model_config_path "{model_config_path}" and model_path "{model_path}" to masking_config.')

    write_to_jsonfile(model_config_path, [('model.path_pre_trained_model', model_path)])
    if return_path:
        return model_config_path
    return json_file_to_pyobj(model_config_path)


def crop_bids_image(resampled_nii_path, crop_values):
    """
    Crop the bids image and save it to resampled_nii_path.

    Parameters
    ----------
    resampled_nii_path :
    crop_values :  values indicating how much the image will be cropped in the x and y direction. Recommended values are [15, 15] and [20, 20].
    """
    resampled_bids_nib = nib.load(resampled_nii_path)
    resampled_bids = resampled_bids_nib.get_data()
    resampled_bids_cropped = resampled_bids[crop_values[0]:resampled_bids.shape[0] - crop_values[1], ...]
    resampled_bids_cropped_nib = nib.Nifti1Image(resampled_bids_cropped, resampled_bids_nib.affine,
                                                 resampled_bids_nib.header)
    nib.save(resampled_bids_cropped_nib, resampled_nii_path)


def get_mask(json_opts, in_file_data, ori_shape, use_cuda: bool):
    """Predict segmentation mask on in_file_data with mlebe model."""
    from mlebe.training.models import get_model
    # To make sure that the GPU is not used for the predictions: (might be unnecessary)
    if not use_cuda:
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


if __name__ == '__main__':
    print(get_biascorrect_opts_defaults({'bias_field_correction': {'bspline_fitting': '[10, 400]'}}))
