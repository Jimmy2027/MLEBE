def predict_mask(
        in_file,
        model_path,
        input_type='anat',
        visualisation_path='',
        visualisation_bool=False,
        visualisation_format='pdf',
        bias_correct_bool=False,
        bias_correct_anat_convergence='[ 150x100x50x30, 1e-16 ]',
        bias_correct_func_convergence='[ 150x100x50x30, 1e-11 ]',
        bias_correct_anat_bspline_fitting='[ 10, 4 ]',
        bias_correct_func_bspline_fitting='[ 10, 4 ]',
        bias_correct_anat_shrink_factor='2',
        bias_correct_func_shrink_factor='2',
        test=False,

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
    # todo this code has to be runable without gpu
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import nibabel as nib
    from mlebe.threed.training.models import get_model
    from mlebe.threed.training.utils.utils import json_file_to_pyobj
    import cv2
    import numpy as np
    from mlebe.threed.training.dataio.transformation import get_dataset_transformation

    def remove_outliers(image):
        """
        Simply counts the number of unconnected objects in the volume and returns the second biggest one (the biggest is the black background)
        """
        from scipy import ndimage
        markers = ndimage.label(image)[0]
        if len(np.unique(markers)) > 2:
            l, counts = np.unique(markers, return_counts=True)
            brain_label = l[np.argsort(-counts)[1]]
            new = np.where(markers == brain_label, 1, 0)
            return new.astype('float64')
        else:
            return image

    json_opts = json_file_to_pyobj(model_path)
    # todo loading the model in samri.pipelines.preprocess gives a memory error
    model = get_model(json_opts.model)
    prediction_shape = json_opts.augmentation.mlebe.scale_size[:2]

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
    """
    Cropping the bids image
    """
    resampled_bids_nib = nib.load(resampled_nii_path)
    resampled_bids = resampled_bids_nib.get_data()
    crop_values = [20, 20]
    resampled_bids_cropped = resampled_bids[crop_values[0]:resampled_bids.shape[0] - crop_values[1], ...]
    resampled_bids_cropped_nib = nib.Nifti1Image(resampled_bids_cropped, resampled_bids_nib.affine,
                                                 resampled_bids_nib.header)
    nib.save(resampled_bids_cropped_nib, resampled_nii_path)

    """
    Bias correction
    """
    if bias_correct_bool == True:
        bias_corrected_path = path.abspath(path.expanduser('corrected_input.nii.gz'))
        if input_type == 'anat':
            command = 'N4BiasFieldCorrection --bspline-fitting {} -d 3 --input-image {} --convergence {} --output {} --shrink-factor {}'.format(
                bias_correct_anat_bspline_fitting, resampled_nii_path, bias_correct_anat_convergence,
                bias_corrected_path, bias_correct_anat_shrink_factor)
        if input_type == 'func':
            command = 'N4BiasFieldCorrection --bspline-fitting {} -d 3 --input-image {} --convergence {} --output {} --shrink-factor {}'.format(
                bias_correct_func_bspline_fitting, resampled_nii_path, bias_correct_func_convergence,
                bias_corrected_path, bias_correct_func_shrink_factor)
        os.system(command)
        print(command)
    else:
        bias_corrected_path = resampled_nii_path

    image = nib.load(bias_corrected_path)
    in_file_data = image.get_data()
    ori_shape = np.moveaxis(in_file_data, 2, 0).shape

    if not test == True:
        # skipping this part for testing as it takes too much time
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
    else:
        mask_pred = np.empty((ori_shape[0], prediction_shape[0], prediction_shape[1]))
        network_input = np.empty((ori_shape[0], prediction_shape[0], prediction_shape[1]))

    # removing predictions that are not attached to the main segmentation in an attempt to remove false segmentation of outer brain regions
    mask_pred = remove_outliers(mask_pred)

    if visualisation_bool == True:
        from matplotlib import pyplot as plt
        save_dir = os.path.join(visualisation_path, os.path.basename(in_file))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # pred_volume_stats(mask_pred, os.path.dirname(os.path.dirname(visualisation_path)), os.path.basename(in_file), model_path)
        for slice in range(in_file_data.shape[0]):
            plt.figure()
            plt.subplot(1, 5, 1)
            plt.imshow(in_file_data[slice], cmap='gray')
            plt.axis('off')
            plt.subplot(1, 5, 2)
            plt.imshow(network_input[slice], cmap='gray')
            plt.title('with dataset transformation')
            plt.axis('off')
            plt.subplot(1, 5, 3)
            plt.imshow(network_input[slice], cmap='gray')
            plt.imshow(mask_pred[slice], cmap='Blues', alpha=0.6)
            plt.axis('off')
            plt.subplot(1, 5, 4)
            plt.imshow(mask_pred[slice])
            plt.axis('off')
            plt.subplot(1, 5, 5)
            plt.imshow(in_file_data[slice], cmap='gray')
            plt.imshow(mask_pred[slice], cmap='Blues', alpha=0.6)
            plt.axis('off')
            plt.savefig(save_dir + '/{}.{}'.format(slice, visualisation_format), format=visualisation_format)
            plt.close()

    """
    Reconstruct to original image size
    """
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
    nib.save(nib.Nifti1Image(resampled_mask_data, input_image.affine, input_image.header), resampled_mask_path)

    masked_image = np.multiply(resampled_mask_data, input_image_data).astype(
        'float32')  # nibabel gives a non-helpful error if trying to save data that has dtype float64
    nii_path_masked = 'masked_output.nii.gz'
    nii_path_masked = path.abspath(path.expanduser(nii_path_masked))
    masked_image = nib.Nifti1Image(masked_image, input_image.affine, input_image.header)
    nib.save(masked_image, nii_path_masked)

    return nii_path_masked, [
        resampled_mask_path], resampled_mask_path  # f/s_biascorrect takes a list as input for the mask while biascorrect takes dierectly the path
