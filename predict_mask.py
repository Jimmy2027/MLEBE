import utils
import numpy as np
import nibabel as nib
import cv2
from tensorflow import keras
import unet

def predict_mask(img):
    prediction_shape = (64, 64)
    img_data = img.get_data()
    img_data = np.moveaxis(img_data, 1, 0)
    ori_shape = img_data.shape
    delta_shape = tuple(np.subtract(prediction_shape, ori_shape[1:]))

    model_path = '/home/hendrik/src/mlebe/new_new_hope0/dice_1200_2019-12-11/1_Step/unet_ep99_val_loss0.28.hdf5'

    model_path = '/Users/Hendrik/Desktop/new_new_hope3/dice_600_2019-12-18/1_Step/unet_ep381_val_loss0.05.hdf5'

    model = keras.models.load_model(model_path, custom_objects={'dice_coef_loss': unet.dice_coef_loss})

    img_data = utils.preprocess(img_data, prediction_shape, 'coronal', switched_axis= True)

    mask_pred = np.empty((ori_shape[0], prediction_shape[0], prediction_shape[1]))
    for slice in range(img_data.shape[0]):
        temp = np.expand_dims(img_data[slice], -1)  # expand dims for channel
        temp = np.expand_dims(temp, 0)  # expand dims for batch
        prediction = model.predict(temp, verbose = 0)
        prediction = np.squeeze(prediction)
        mask_pred[slice, ...] = prediction


    """
    Reconstruct to original image size 
    """
    resized = np.empty(ori_shape)
    for i, slice in enumerate(mask_pred):

        if delta_shape[0] < 0 and delta_shape[0] < 0:
            resized_mask = cv2.resize(slice, (ori_shape[2], ori_shape[1]))
            resized[i] = resized_mask
        elif delta_shape[0] < 0:
            temp = cv2.resize(slice, (prediction_shape[1]), ori_shape[1])
            resized_mask = temp[:, delta_shape[1]//2:ori_shape[2] + delta_shape[1]//2]
            resized[i] = resized_mask
        elif delta_shape[1] < 0:
            temp = cv2.resize(slice, (ori_shape[2], prediction_shape[0]))
            resized_mask = temp[delta_shape[0]//2:ori_shape[1] + delta_shape[0]//2, :]
            resized[i] = resized_mask
        elif delta_shape[0] < 0 and delta_shape[1] < 0:
            resized_mask = slice[delta_shape[0]//2:ori_shape[1] + delta_shape[0]//2, delta_shape[1]//2:ori_shape[2] + delta_shape[1]//2]
            resized[i] = resized_mask

    resized = np.moveaxis(resized, 1, 0)
    masked_image = img.get_data()

    masked_image[resized == 0] = 0




    masked_image = nib.Nifti1Image(masked_image, img.affine)


    return masked_image




