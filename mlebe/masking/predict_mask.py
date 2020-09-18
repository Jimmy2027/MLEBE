def predict_mask(
        in_file,
        workflow_config_path,
        input_type='anat',
):
    """
    The image is first resampled into the resolution of the template space, which has a voxel size of 0.2 × 0.2 × 0.2. This is done with the Resample command from the FSL library which is an analysis tool for FMRI, MRI and DTI brain imaging data. Then, the image is preprocessed using the preprocessing methods of the model class. The predictions of the model are reconstructed to a 3D mask via the command Nifit1Image from nibabel. This is done using the same affine space as the input image. The latter is then reshaped into the original shape inverting the preprocessing step, either with the opencv resize method or by cropping. Additionally, the binary mask is resampled into its original affine space, before being multiplied with the brain image to extract the ROI.

    Parameters
    ----------
    in_file : str
        path to the file that is to be masked
    workflow_config_path : str
        path to the workflow config. The workflow config is a json file that must contain the path
        to the model json config file. All other parameters have default values that will be set
        in the "get_masking_opts" method.
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
    from mlebe.masking.utils import get_mask

    masking_opts = get_masking_opts(workflow_config_path, input_type)
    model_config = get_model_config(masking_opts)
    input = in_file
    if input_type == 'func':
        tMean_path = 'tMean.nii.gz'
        command = 'fslmaths {a} -Tmean {b}'.format(a=input, b=tMean_path)
        print(command)
        os.system(command)
        input = tMean_path

    resampled_path = 'resampled_input.nii.gz'
    resampled_nii_path = path.abspath(path.expanduser(resampled_path))
    resample_cmd = 'ResampleImage 3 {input} '.format(input=input) + resampled_nii_path + ' 0.2x0.2x0.2'
    os.system(resample_cmd)
    print(resample_cmd)

    if masking_opts['with_bids_cropping']:
        crop_bids_image(resampled_nii_path, masking_opts['crop_values'])

    """
    Bias correction
    """
    if masking_opts['bias_correct_bool'] == True:
        bias_correction_config = masking_opts['bias_field_correction']
        bias_corrected_path = path.abspath(path.expanduser('corrected_input.nii.gz'))
        if input_type == 'anat':
            command = 'N4BiasFieldCorrection --bspline-fitting {} -d 3 --input-image {} --convergence {} --output {} --shrink-factor {}'.format(
                bias_correction_config['bspline_fitting'], resampled_nii_path,
                bias_correction_config['convergence'],
                bias_corrected_path, bias_correction_config['shrink_factor'])
        if input_type == 'func':
            command = 'N4BiasFieldCorrection --bspline-fitting {} -d 3 --input-image {} --convergence {} --output {} --shrink-factor {}'.format(
                bias_correction_config['bspline_fitting'], resampled_nii_path,
                bias_correction_config['convergence'],
                bias_corrected_path, bias_correction_config['shrink_factor'])
        os.system(command)
        print(command)
    else:
        bias_corrected_path = resampled_nii_path

    image = nib.load(bias_corrected_path)
    in_file_data = image.get_data()

    """
    Getting the mask
    """

    if not masking_opts['test'] == True:
        ori_shape = np.moveaxis(in_file_data, 2, 0).shape
        in_file_data, mask_pred, network_input = get_mask(model_config, in_file_data, ori_shape)
    else:
        prediction_shape = (128, 128)
        ori_shape = np.moveaxis(in_file_data, 2, 0).shape
        mask_pred = np.empty((ori_shape[0], prediction_shape[0], prediction_shape[1]))
        network_input = np.empty((ori_shape[0], prediction_shape[0], prediction_shape[1]))

    mask_pred = remove_outliers(mask_pred)

    if masking_opts['visualisation_bool'] == True:
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
    print(resample_cmd)
    os.system(resample_cmd)

    resampled_mask = nib.load(resampled_mask_path)
    resampled_mask_data = resampled_mask.get_data()
    input_image_data = input_image.get_data()
    if not resampled_mask_data.shape == input_image_data.shape:
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

    return nii_path_masked, [
        resampled_mask_path], resampled_mask_path  # f/s_biascorrect takes a list as input for the mask while biascorrect takes dierectly the path
