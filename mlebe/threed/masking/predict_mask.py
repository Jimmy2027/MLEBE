def predict_mask(
        in_file,
        workflow_config_path,
        input_type='anat',
):
    """
    :param in_file: path to the file that is to be masked
    :param input_type: either 'func' for CDV or BOLD contrast or 'anat' for T2 contrast
    :param visualisation: dictionary with
        'bool': indicates if the predictions will be saved for visualisation
        'path': path where the visualisations will be saved
    :param model_path: path to trained model for the masking (can be downloaded here: https://zenodo.org/record/3759361#.XqBhyVMzZhH)
    if using 3D model, this is the path to the config file of that model
    :return:
    """

    import os
    from os import path
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import nibabel as nib
    from mlebe.threed.training.models import get_model
    from mlebe.threed.training.utils.utils import json_file_to_pyobj
    import numpy as np
    from mlebe.threed.masking.utils import remove_outliers, get_workflow_config, crop_bids_image, \
        get_mask, save_visualisation, reconstruct_image, pad_to_shape

    workflow_config = get_workflow_config(workflow_config_path, input_type)
    model_json_opts = json_file_to_pyobj(workflow_config.model_config_path)
    # todo loading the model in samri.pipelines.preprocess gives a memory error
    model = get_model(model_json_opts.model)
    prediction_shape = model_json_opts.augmentation.mlebe.scale_size[:2]

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

    if workflow_config.with_bids_cropping:
        crop_bids_image(resampled_nii_path)

    """
    Bias correction
    """
    if workflow_config.bias_correct_bool == True:
        bias_correction_config = workflow_config.bias_field_correction
        bias_corrected_path = path.abspath(path.expanduser('corrected_input.nii.gz'))
        if input_type == 'anat':
            command = 'N4BiasFieldCorrection --bspline-fitting {} -d 3 --input-image {} --convergence {} --output {} --shrink-factor {}'.format(
                bias_correction_config.bspline_fitting, resampled_nii_path,
                bias_correction_config.convergence,
                bias_corrected_path, bias_correction_config.shrink_factor)
        if input_type == 'func':
            command = 'N4BiasFieldCorrection --bspline-fitting {} -d 3 --input-image {} --convergence {} --output {} --shrink-factor {}'.format(
                bias_correction_config.bspline_fitting, resampled_nii_path,
                bias_correction_config.convergence,
                bias_corrected_path, bias_correction_config.shrink_factor)
        os.system(command)
        print(command)
    else:
        bias_corrected_path = resampled_nii_path

    image = nib.load(bias_corrected_path)
    in_file_data = image.get_data()
    ori_shape = np.moveaxis(in_file_data, 2, 0).shape

    if not workflow_config.test == True:
        # skipping this part for testing as it takes too much time
        in_file_data, mask_pred, network_input = get_mask(model_json_opts, in_file_data, model, ori_shape)
    else:
        mask_pred = np.empty((ori_shape[0], prediction_shape[0], prediction_shape[1]))
        network_input = np.empty((ori_shape[0], prediction_shape[0], prediction_shape[1]))

    # removing predictions that are not attached to the main segmentation in an attempt to remove false segmentation of outer brain regions
    mask_pred = remove_outliers(mask_pred)

    if workflow_config.visualisation_bool == True:
        save_visualisation(workflow_config, in_file, in_file_data, network_input, mask_pred)

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
        resampled_mask_path], resampled_mask_path  # f/s_biascorrect takes a list as input for the mask while biascorrect takes directly the path
