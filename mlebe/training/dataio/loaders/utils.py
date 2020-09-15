import os
import nibabel as nib
import scipy
import numpy as np


def validate_images(image, label=None):
    if label is not None:
        if image.shape[:-1] != label.shape[:-1]:
            print('Error: mismatched size, image.shape = {0}, '
                  'label.shape = {1}'.format(image.shape, label.shape))
            raise (Exception('image and label sizes do not match'))

    if image.max() < 1e-6:
        print('Error: blank image, image.max = {0}'.format(image.max()))
        raise (Exception('blank image exception'))


def load_mask(data_dir):
    for o in os.listdir(data_dir):
        if o == 'dsurqec_200micron_mask.nii':
            mask_path = os.path.join(data_dir, o)

    mask = nib.load(mask_path)
    return mask


def arrange_mask(img, mask):
    """
    Sets the mask to zero where the image is zero
    """

    new_mask = mask[:, :, :]

    new_mask[img == 0] = 0

    fixed_mask = new_mask[:, :, :]

    structure = [[1, 0, 1], [1, 1, 1], [0, 1, 0]]

    for i in range(new_mask.shape[0]):
        fixed_mask[i] = scipy.ndimage.morphology.binary_fill_holes(new_mask[i], structure=structure)

    return fixed_mask


def write_blacklist(blacklist_dir):
    blacklist = []

    class blacklist_elem:
        def __init__(self, subject, session):
            self.subj = subject
            self.sess = session

    for file in os.listdir(blacklist_dir):
        if file.endswith('.pdf'):
            temp = file.replace('.pdf', '')
            temp2 = temp.split('_')
            blacklist.append(blacklist_elem(temp2[0], temp2[1]))
    return blacklist


def remove_black_images(img, mask, test=False):
    new_img = img[:, :, :]
    new_mask = mask[:, :, :]

    counter = 0
    # when test == true image is a np.empty -- this function would remove all slices so it is skipped for testing
    if not img.shape[0] == 0 and not test == True:
        for z in range(img.shape[0]):

            if len(np.unique(img[z, ...])) == 1:
                new_img = np.delete(new_img, z - counter, 0)
                new_mask = np.delete(new_mask, z - counter, 0)
                counter += 1

        assert new_img.shape[
                   0] != 0, 'all slices have been removed by "remove_black_images", old_shape: {}, new_shape: {}'.format(
            img.shape, new_img.shape)

    return new_img, new_mask


def data_normalization(data):
    """
    :param data: shape: (y, x)
    :return: normalised input
    """
    data = data * 1.
    data = np.clip(data, 0, np.percentile(data, 99))

    data = data - np.amin(data)
    if np.amax(data) != 0:
        data = data / np.amax(data)
    return data
