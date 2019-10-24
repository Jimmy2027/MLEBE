import os
import imageio
import numpy as np
from matplotlib import pyplot as plt
import os

# from ..core import ants_image as iio
# from .. import utils
#
#
# def resample_image_to_target(image, target, interp_type='linear', imagetype=0, verbose=False, **kwargs):
#     """
#     Resample image by using another image as target reference.
#     This function uses ants.apply_transform with an identity matrix
#     to achieve proper resampling.
#
#     ANTsR function: `resampleImageToTarget`
#
#     Arguments
#     ---------
#     image : ANTsImage
#         image to resample
#
#     target : ANTsImage
#         image of reference, the output will be in this space
#
#     interp_type : string
#         Choice of interpolator. Supports partial matching.
#             linear
#             nearestNeighbor
#             multiLabel for label images but genericlabel is preferred
#             gaussian
#             bSpline
#             cosineWindowedSinc
#             welchWindowedSinc
#             hammingWindowedSinc
#             lanczosWindowedSinc
#             genericLabel use this for label images
#
#     imagetype : integer
#         choose 0/1/2/3 mapping to scalar/vector/tensor/time-series
#
#     verbose : boolean
#         print command and run verbose application of transform.
#
#     kwargs : keyword arguments
#         additional arugment passed to antsApplyTransforms C code
#
#     Returns
#     -------
#     ANTsImage
#
#     Example
#     -------
#     >>> import ants
#     >>> fi = ants.image_read(ants.get_ants_data('r16'))
#     >>> fi2mm = ants.resample_image(fi, (2,2), use_voxels=0, interp_type='linear')
#     >>> resampled = ants.resample_image_to_target(fi2mm, fi, verbose=True)
#     """
#     fixed = target
#     moving = image
#     compose = None
#     transformlist = 'identity'
#     interpolator = interp_type
#
#     interpolator_oldoptions = ("linear", "nearestNeighbor", "gaussian", "cosineWindowedSinc", "bSpline")
#     if isinstance(interp_type, int):
#         interpolator = interpolator_oldoptions[interp_type]
#
#     accepted_interpolators = {"linear", "nearestNeighbor", "multiLabel", "gaussian",
#                               "bSpline", "cosineWindowedSinc", "welchWindowedSinc",
#                               "hammingWindowedSinc", "lanczosWindowedSinc", "genericLabel"}
#
#     if interpolator not in accepted_interpolators:
#         raise ValueError('interpolator not supported - see %s' % accepted_interpolators)
#
#     args = [fixed, moving, transformlist, interpolator]
#
#     if not isinstance(fixed, str):
#         if isinstance(fixed, iio.ANTsImage) and isinstance(moving, iio.ANTsImage):
#             inpixeltype = fixed.pixeltype
#             warpedmovout = moving.clone()
#             f = fixed
#             m = moving
#             if (moving.dimension == 4) and (fixed.dimension == 3) and (imagetype == 0):
#                 raise ValueError('Set imagetype 3 to transform time series images.')
#
#             wmo = warpedmovout
#             mytx = ['-t', 'identity']
#             if compose is None:
#                 args = ['-d', fixed.dimension, '-i', m, '-o', wmo, '-r', f, '-n', interpolator] + mytx
#
#             tfn = '%scomptx.nii.gz' % compose if compose is not None else 'NA'
#             if compose is not None:
#                 mycompo = '[%s,1]' % tfn
#                 args = ['-d', fixed.dimension, '-i', m, '-o', mycompo, '-r', f, '-n', interpolator] + mytx
#
#             myargs = utils._int_antsProcessArguments(args)
#
#             # NO CLUE WHAT THIS DOES OR WHY IT'S NEEDED
#             for jj in range(len(myargs)):
#                 if myargs[jj] is not None:
#                     if myargs[jj] == '-':
#                         myargs2 = [None] * (len(myargs) - 1)
#                         myargs2[:(jj - 1)] = myargs[:(jj - 1)]
#                         myargs2[jj:(len(myargs) - 1)] = myargs[(jj + 1):(len(myargs))]
#                         myargs = myargs2
#
#             myverb = int(verbose)
#
#             processed_args = myargs + ['-z', str(1), '-v', str(myverb), '--float', str(1), '-e', str(imagetype)]
#             libfn = utils.get_lib_fn('antsApplyTransforms')
#             libfn(processed_args)
#
#             if compose is None:
#                 return warpedmovout.clone(inpixeltype)
#             else:
#                 if os.path.exists(tfn):
#                     return tfn
#                 else:
#                     return None
#         else:
#             return 1
#     else:
#         processed_args = myargs + ['-z', str(1), '--float', str(1), '-e', str(imagetype)]
#         libfn = utils.get_lib_fn('antsApplyTransforms')
#         libfn(processed_args)

def data_normalization(data):
    """

    :param data: shape: (x,y)
    :return: normalised input
    """

    for i in data:
        for j in range(0, i.shape[0]):
            i[j] = i[j]*1.
            i[j] = np.clip(i[j], 0, np.percentile(i[j], 99))

            i[j] = i[j] - np.amin(i[j])
            if np.amax(i[j]) != 0:
                i[j] = i[j] / np.amax(i[j])
    return data


def save_img(img_data, path):
    if not os.path.exists(path):
        os.makedirs(path)
    for j in range(img_data.shape[0]):
        print(img_data[j, ...].shape)
        plt.imshow(img_data[j, ...], cmap='gray')
        plt.savefig(os.path.join(path, 'img_{}.png'.format(j)))



def save_datavisualisation3(img_data, myocar_labels, predicted_labels, save_folder, index_first = False, normalized = False):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    img_data_temp = []
    myocar_labels_temp = []
    predicted_labels_temp = []
    if index_first == True:
        for i in range(0, len(img_data)):
            img_data_temp.append(np.moveaxis(img_data[i], 0, -1))
            myocar_labels_temp.append(np.moveaxis(myocar_labels[i], 0, -1))
            predicted_labels_temp.append(np.moveaxis(predicted_labels[i], 0, -1))
    counter = 0
    for i, j, k in zip(img_data_temp[:], myocar_labels_temp[:], predicted_labels_temp[:]):

        i_patch = i[:, :, 0]
        if normalized == True:
            i_patch = i_patch*255
        # np.squeeze(i_patch)

        j_patch = j[:, :, 0]
        # np.squeeze(j_patch)
        j_patch = j_patch * 255

        k_patch = k[:,:,0]
        k_patch = k_patch*255

        for slice in range(1, i.shape[2]):
            temp = i[:, :, slice]
            # np.squeeze(temp)
            if normalized == True:
                temp = temp * 255
            i_patch = np.hstack((i_patch, temp))


            temp = j[:, :, slice]
            # np.squeeze(temp)
            temp = temp * 255
            j_patch = np.hstack((j_patch, temp))

            temp = k[:,:,slice]
            temp = temp*255
            k_patch = np.hstack((k_patch, temp))

        image = np.vstack((i_patch, j_patch, k_patch))

        imageio.imwrite(save_folder + 'mds' + '%d.png' % (counter,), image)


        counter = counter + 1