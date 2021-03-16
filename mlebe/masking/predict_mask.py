# -*- coding: utf-8 -*-

"""Masking function of the mlebe package."""

from typing import Optional


def predict_mask(
        in_file: str,
        masking_config_path: Optional[str] = None,
        input_type: str = 'anat',
):
    """
    The image is first resampled into the resolution of the template space, which has a voxel size of 0.2 × 0.2 × 0.2.
    This is done with the Resample command from the FSL library which is an analysis tool for FMRI, MRI and DTI brain
    imaging data. Then, the image is preprocessed using the preprocessing methods of the model class.
    The predictions of the model are reconstructed to a 3D mask via the command Nifit1Image from nibabel.
    This is done using the same affine space as the input image. The latter is then reshaped into the original shape
    inverting the preprocessing step, either with the opencv resize method or by cropping.
    Additionally, the binary mask is resampled into its original affine space, before being multiplied with the brain
    image to extract the ROI.

    Parameters
    ----------
    in_file : str
        path to the file that is to be masked
    masking_config_path : str
        path to the masking config. The masking config is a json file. All parameters have default values that will
        be set in the "get_masking_opts" method.
        The masking config may contain following parameters (if any of them is not given in the config,
        the default value will be taken from mlebe/masking/config/default_schema.json):
        model_folder_path: str The path to the pretrained model. If not set the default mlebe model will be selected.
        use_cuda: bool
            boolean indicating if cuda will be used for the masking
        visualisation_path: str
            if set, the masking predictions will be saved to this destination.
        crop_values:
            if set, the input bids images will be cropped with given values in the x-y dimensions.
        bias_field_correction: dict
            If set, the input image will be bias corrected before given as input to the model.
            The parameter of the bias correction can be given as a dictionary.
            The default values can be found in the default_schema.json config.



    input_type : str
        either 'func' for CDV or BOLD contrast or 'anat' for T2 contrast

    Returns
    -------
    resampled_mask_path : str
        path to the mask
    nii_path_masked : str
        path to the masked image
    """
    import os
    from os import path
    import nibabel as nib
    import numpy as np
    from mlebe.masking.utils import remove_outliers, get_masking_opts, crop_bids_image, \
        save_visualisation, reconstruct_image, pad_to_shape, get_model_config
    from mlebe.masking.utils import get_mask, get_mlebe_models, get_biascorrect_opts_defaults
    from mlebe import log

    masking_opts = get_masking_opts(masking_config_path, input_type)
    if 'model_folder_path' not in masking_opts or not masking_opts['model_folder_path']:
        # if no model_folder_path is given in the config, the default models are selected.
        masking_opts['model_folder_path'] = get_mlebe_models(input_type)
    model_config = get_model_config(masking_opts)
    input = in_file
    if input_type == 'func':
        tMean_path = 'tMean.nii.gz'
        command = 'fslmaths {a} -Tmean {b}'.format(a=input, b=tMean_path)
        log.info(f'Executing command "{command}"')
        os.system(command)
        input = tMean_path

    resampled_path = 'resampled_input.nii.gz'
    resampled_nii_path = path.abspath(path.expanduser(resampled_path))
    resample_cmd = 'ResampleImage 3 {input} '.format(input=input) + resampled_nii_path + ' 0.2x0.2x0.2'
    os.system(resample_cmd)
    log.info(f'Resample image with "{resample_cmd}"')

    if 'crop_values' in masking_opts and masking_opts['crop_values']:
        crop_bids_image(resampled_nii_path, masking_opts['crop_values'])

    """
    Bias correction
    """
    if 'bias_field_correction' in masking_opts and masking_opts['bias_field_correction']:
        bias_correction_config = get_biascorrect_opts_defaults(masking_opts)
        bias_corrected_path = path.abspath(path.expanduser('corrected_input.nii.gz'))

        command = 'N4BiasFieldCorrection --bspline-fitting {} -d 3 --input-image {} --convergence {} --output {} --shrink-factor {}'.format(
            bias_correction_config['bspline_fitting'], resampled_nii_path,
            bias_correction_config['convergence'],
            bias_corrected_path, bias_correction_config['shrink_factor'])

        os.system(command)
        log.info(f'Apply bias correction with "{command}"')

    else:
        bias_corrected_path = resampled_nii_path

    image = nib.load(bias_corrected_path)
    in_file_data = image.get_data()

    """
    Getting the mask
    """
    ori_shape = np.moveaxis(in_file_data, 2, 0).shape
    in_file_data, mask_pred, network_input = get_mask(model_config, in_file_data, ori_shape,
                                                      use_cuda=masking_opts['use_cuda'])

    mask_pred = remove_outliers(mask_pred)

    if 'visualisation_path' in masking_opts and masking_opts['visualisation_path']:
        save_visualisation(masking_opts, in_file, network_input, mask_pred)

    """
    Reconstruct to original image size
    """
    resized = reconstruct_image(ori_shape, mask_pred)

    resized_path = 'resized_mask.nii.gz'
    resized_path = path.abspath(path.expanduser(resized_path))
    resized_mask = nib.Nifti1Image(resized, image.affine, image.header)
    nib.save(resized_mask, resized_path)

    input_image = nib.load(input)
    input_img_affine = input_image.affine
    voxel_sizes = nib.affines.voxel_sizes(input_img_affine)

    resampled_mask_path = 'resampled_mask.nii.gz'
    resampled_mask_path = path.abspath(path.expanduser(resampled_mask_path))
    resample_cmd = 'ResampleImage 3 {input} '.format(
        input=resized_path) + ' ' + resampled_mask_path + ' {x}x{y}x{z} '.format(x=voxel_sizes[0], y=voxel_sizes[1],
                                                                                 z=voxel_sizes[2]) + ' 0 1'
    log.info(f'Resample image with "{resample_cmd}"')
    os.system(resample_cmd)

    resampled_mask = nib.load(resampled_mask_path)
    resampled_mask_data = resampled_mask.get_data()
    input_image_data = input_image.get_data()
    if resampled_mask_data.shape != input_image_data.shape:
        resampled_mask_data = pad_to_shape(resampled_mask_data, input_image_data)
    nib.save(nib.Nifti1Image(resampled_mask_data, input_image.affine, input_image.header), resampled_mask_path)

    """
    Masking of the input image
    """
    masked_image = np.multiply(resampled_mask_data, input_image_data).astype(
        'float32')  # nibabel gives a non-helpful error if trying to save data that has dtype float64
    nii_path_masked = 'masked_output.nii.gz'
    nii_path_masked = path.abspath(path.expanduser(nii_path_masked))
    masked_image = nib.Nifti1Image(masked_image, input_image.affine, input_image.header)
    nib.save(masked_image, nii_path_masked)

    log.info(f'Masking of input image {in_file} finished successfully.')
    # f/s_biascorrect takes a list as input for the mask while biascorrect takes directly the path
    return nii_path_masked, [resampled_mask_path], resampled_mask_path


if __name__ == '__main__':
    nii_path_masked, something, aha = predict_mask(in_file=
                                                   '/home/hendrik/.scratch/mlebe/bids/sub-4012/ses-ofMpF/func/sub-4012_ses-ofMpF_task-JogB_acq-EPIlowcov_run-0_bold.nii.gz',
                                                   input_type='func',
                                                   # masking_config_path='/home/hendrik/.scratch/mlebe/config.json'
                                                   )
