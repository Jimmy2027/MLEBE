import os
from pathlib import Path
from typing import Iterable, Optional

import nibabel as nib
import numpy as np
import pandas as pd
import scipy


def make_dataselection_anat(data_dir: Path, studies: Iterable[str], blacklist=None, save_dir: Optional[Path] = None,
                            split: str = None):
    data_selection = pd.DataFrame()
    blacklist_selection = pd.DataFrame()
    for o in os.listdir(data_dir):
        if (not studies or o in studies) and not o.startswith('.') and not o.endswith(
                '.xz'):
            print(o)
            data_set = o
            for x in os.listdir(os.path.join(data_dir, o)):
                if (x.endswith('preprocessed') or x.startswith('preprocess') or x.endswith(
                        'preprocessing')) and not x.endswith('work'):
                    for root, dirs, files in os.walk(os.path.join(data_dir, o, x)):
                        for file in files:
                            if not file.startswith('.') and (
                                    file.endswith("_T2w.nii.gz") or file.endswith("_T1w.nii.gz")):
                                split = file.split('_')
                                subject = split[0].split('-')[1]
                                session = split[1].split('-')[1]
                                acquisition = split[2].split('-')[1]
                                type = split[3].split('.')[0]
                                uid = file.split('.')[0]
                                path = os.path.join(root, file)
                                blacklisted = False
                                if blacklist:
                                    for i in blacklist:
                                        if subject == i.subj and session == i.sess:
                                            blacklisted = True
                                            blacklist_selection = pd.concat([blacklist_selection, pd.DataFrame(
                                                [[data_set, subject, session, acquisition, type, uid, path]],
                                                columns=['data_set', 'subject', 'session', 'acquisition', 'type',
                                                         'uid',
                                                         'path'])]).reset_index(drop=True)
                                if not blacklisted:
                                    data_selection = pd.concat([data_selection, pd.DataFrame(
                                        [[data_set, subject, session, acquisition, type, uid, path]],
                                        columns=['data_set', 'subject', 'session', 'acquisition', 'type', 'uid',
                                                 'path'])]).reset_index(drop=True)
    if save_dir:
        data_selection.to_csv(save_dir / f'{split}_dataset.csv', index=False)

    return data_selection, blacklist_selection


def make_dataselection_func(data_dir, studies, func_training_dir: Path, save_dir: Path = None, split: str = None):
    """
    Create a data selection dataframe for the functional data. The functional scans are collapsed over time using
    fslmaths -Tmean and saved to func_training_dir, if they are not already present there.
    Parameters
    ----------
    data_dir :
    studies :
    func_training_dir :
    save_dir :
    split :

    Returns
    -------

    """
    data_selection = pd.DataFrame()

    if not os.path.exists(func_training_dir):
        print('creating dir: ', func_training_dir)
        os.makedirs(func_training_dir)
    for o in os.listdir(data_dir):
        if o in studies and not o.startswith('.') and not o.startswith('.') and not o.endswith('.xz'):
            data_set = o
            for x in os.listdir(os.path.join(data_dir, o)):
                if (x.endswith('preprocessed') or x.startswith('preprocess') or x.endswith(
                        'preprocessing')) and not x.endswith('work'):
                    for root, dirs, files in os.walk(os.path.join(data_dir, o, x)):
                        if root.endswith('func'):
                            for file in files:
                                if file.endswith(".nii.gz"):
                                    tMean_path = os.path.join(func_training_dir, 'tMean_' + file)
                                    # collapse volumes over time
                                    if not os.path.isfile(tMean_path):
                                        command = 'fslmaths {a} -Tmean {b}'.format(a=os.path.join(root, file),
                                                                                   b=tMean_path)
                                        print(command)
                                        os.system(command)

                                    split = file.split('_')
                                    subject = split[0].split('-')[1]
                                    session = split[1].split('-')[1]
                                    acquisition = split[2].split('-')[1]
                                    type = split[3].split('.')[0]
                                    uid = file.split('.')[0]
                                    path = tMean_path
                                    data_selection = pd.concat([data_selection, pd.DataFrame(
                                        [[data_set, subject, session, acquisition, type, uid, path]],
                                        columns=['data_set', 'subject', 'session', 'acquisition', 'type', 'uid',
                                                 'path'])])
    if save_dir:
        data_selection.to_csv(os.path.join(save_dir, split + '_dataset.csv'), index=False)
    assert len(data_selection.data_set.unique()) == len(studies), 'Only found {} studies, expected {}'.format(
        data_selection.data_set.unique(), studies)
    return data_selection


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
