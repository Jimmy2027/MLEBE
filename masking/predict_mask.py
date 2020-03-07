def predict_mask(in_file, input_type = 'anat'):
    import os
    from os import path
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import nibabel as nib
    from samri.masking import utils
    import numpy as np
    import cv2
    from tensorflow import keras
    import tensorflow.keras.backend as K

    def dice_coef(y_true, y_pred, smooth=1):
        """
        Dice = (2*|X & Y|)/ (|X|+ |Y|)
             =  2*sum(|A*B|)/(sum(A)+sum(B))
        ref: https://arxiv.org/pdf/1606.04797v1.pdf
        """

        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)

        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def dice_coef_loss(y_true, y_pred):
        return 1 - dice_coef(y_true, y_pred)

    def remove_outliers(image):
        from scipy import ndimage
        from skimage.morphology import watershed
        import os
        from matplotlib import pyplot as plt
        markers = ndimage.label(image)[0]
        if len(np.unique(markers)) > 2:
            if not os.path.exists(path.abspath(path.expanduser('classifier/'))):
                os.makedirs(path.abspath(path.expanduser('classifier/')))
            l, counts = np.unique(markers, return_counts=True)
            brain_label = l[np.argsort(-counts)[1]]
            new = np.where(markers == brain_label, 1, 0)
            return new.astype('float64')
        else:
            return image
    prediction_shape = (128, 128)
    if input_type == 'func':
        tMean_path = 'tMean.nii.gz'
        command = 'fslmaths {a} -Tmean {b}'.format(a=in_file, b=tMean_path)
        print(command)
        os.system(command)
        in_file = tMean_path

    resampled_path = 'resampled_input.nii.gz'
    resampled_nii_path = path.abspath(path.expanduser(resampled_path))
    resample_cmd = 'ResampleImage 3 {input} '.format(input=in_file) + resampled_nii_path + ' 0.2x0.2x0.2'
    os.system(resample_cmd)
    print(resample_cmd)

    image = nib.load(resampled_nii_path)
    in_file_data = image.get_data()
    in_file_data = np.moveaxis(in_file_data, 2, 0)
    ori_shape = in_file_data.shape

    if input_type == 'anat':
        model_path = '/mnt/data/mlebe_data/results/new_bl128/dice_600_2020-02-07/1_Step/model_ep483.h5'
    if input_type == 'func':
        model_path = '/mnt/data/mlebe_data/results/func_training/dice_600_2020-02-21/1_Step/model_ep86.h5'

    model = keras.models.load_model(model_path, custom_objects={'dice_coef_loss': dice_coef_loss})
    in_file_data = utils.preprocess(in_file_data, prediction_shape, 'coronal', switched_axis= True)

    mask_pred = np.empty((ori_shape[0], prediction_shape[0], prediction_shape[1]))
    for slice in range(in_file_data.shape[0]):
        temp = np.expand_dims(in_file_data[slice], -1)  # expand dims for channel
        temp = np.expand_dims(temp, 0)  # expand dims for batch
        prediction = model.predict(temp, verbose = 0)
        prediction = np.squeeze(prediction)
        mask_pred[slice, ...] = np.where(prediction > 0.9, 1, 0)

    mask_pred = remove_outliers(mask_pred)

    """
    Reconstruct to original image size
    """
    resized = np.empty(ori_shape)
    for i, slice in enumerate(mask_pred):
        if ori_shape[1] < ori_shape[2]:
            padd = ori_shape[2] - ori_shape[1]
            resized_mask_temp = cv2.resize(slice, (ori_shape[2],ori_shape[2]))
            resized_mask = resized_mask_temp[padd//2:ori_shape[1] + padd//2, :]

            resized[i] = resized_mask
        elif ori_shape[1] > ori_shape[2]:
            padd = ori_shape[1] - ori_shape[2]
            resized_mask_temp = cv2.resize(slice, (ori_shape[1], ori_shape[1]))

            resized_mask = resized_mask_temp[:, padd//2:ori_shape[2] + padd//2]
            resized[i] = resized_mask
        else:
            resized_mask = cv2.resize(slice, (ori_shape[2], ori_shape[1]))
            resized[i] = resized_mask

    resized = np.moveaxis(resized, 0, 2)

    resized_path = 'resized_mask.nii.gz'
    resized_path = path.abspath(path.expanduser(resized_path))
    resized_mask = nib.Nifti1Image(resized, image.affine, image.header)
    nib.save(resized_mask, resized_path)

    input_image = nib.load(in_file)
    input_img_affine = input_image.affine
    voxel_sizes = nib.affines.voxel_sizes(input_img_affine)

    resampled_mask_path = 'resampled_mask.nii.gz'
    resampled_mask_path = path.abspath(path.expanduser(resampled_mask_path))
    resample_cmd = 'ResampleImage 3 {input} '.format(input=resized_path) + ' ' + resampled_mask_path + ' {x}x{y}x{z} '.format(x= voxel_sizes[0], y= voxel_sizes[1], z = voxel_sizes[2]) + ' 0 1'
    print(resample_cmd)
    os.system(resample_cmd)

    if input_type == 'anat':
        resampled_mask = nib.load(resampled_mask_path)
        resampled_mask_data = resampled_mask.get_data()
        input_image_data = input_image.get_data()
        masked_image = np.multiply(resampled_mask_data, input_image_data)

        nii_path_masked = 'masked_output.nii.gz'
        nii_path_masked = path.abspath(path.expanduser(nii_path_masked))
        masked_image = nib.Nifti1Image(masked_image, input_image.affine, input_image.header)
        nib.save(masked_image, nii_path_masked)

    if input_type == 'func':
        return resampled_mask_path
    else:
        return nii_path_masked, [resampled_mask_path], resampled_mask_path  #f/s_biascorrect takes a list as input for the mask while biascorrect takes dierectly the path
