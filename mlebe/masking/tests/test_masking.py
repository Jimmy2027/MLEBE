import json
import os
import shutil

import nibabel as nib
import numpy as np

import mlebe
from mlebe.masking.predict_mask import predict_mask


def test_masker():
    test_dir = os.path.join(os.path.dirname(mlebe.__file__), 'masking/tests/temp/')
    os.makedirs(test_dir, exist_ok=True)
    test_config = {"masking_config": {"masking_config_anat": {"test": True}}}
    with open(os.path.join(test_dir, 'test_config.json'), 'w') as jsonfile:
        json.dump(test_config, jsonfile, indent=4)
    test_in_file = np.ones((63, 96, 48))
    test_in_file = nib.Nifti1Image(test_in_file, np.eye(4))
    test_in_file_dir = os.path.join(test_dir, 'test_in_file.nii.gz')
    nib.save(test_in_file, test_in_file_dir)
    _, _, _ = predict_mask(test_in_file_dir, os.path.join(test_dir, 'test_config.json'))
    shutil.rmtree(test_dir)


if __name__ == '__name__':
    test_masker()
