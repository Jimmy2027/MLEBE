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
    :return:
    """

    import os
    from os import path
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import nibabel as nib
    from mlebe.training.utils import general
    import numpy as np
    from tensorflow import keras
    from mlebe.training.unet import dice_coef, dice_coef_loss
    from mlebe.threed.masking.utils import remove_outliers, get_workflow_config, crop_bids_image, \
        save_visualisation, reconstruct_image, pad_to_shape
    import pandas as pd

    workflow_config = get_workflow_config(workflow_config_path, input_type)
    model_config = pd.read_csv(workflow_config.model_config_path).iloc[0].to_dict()
    prediction_shape = (128, 128)
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
    in_file_data = np.moveaxis(in_file_data, 2, 0)
    ori_shape = in_file_data.shape

    if not workflow_config.test == True:
        model = keras.models.load_model(model_config['model_path'],
                                        custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    in_file_data = general.preprocess(in_file_data, prediction_shape, 'coronal', switched_axis=True)

    mask_pred = np.empty((ori_shape[0], prediction_shape[0], prediction_shape[1]))

    if not workflow_config.test == True:
        for slice in range(in_file_data.shape[0]):
            temp = np.expand_dims(in_file_data[slice], -1)  # expand dims for channel
            temp = np.expand_dims(temp, 0)  # expand dims for batch
            prediction = model.predict(temp, verbose=0)
            prediction = np.squeeze(prediction)
            mask_pred[slice, ...] = np.where(prediction > 0.9, 1, 0)

    mask_pred = remove_outliers(mask_pred)

    if workflow_config.visualisation_bool == True:
        save_visualisation(workflow_config, in_file, in_file_data, mask_pred)

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
