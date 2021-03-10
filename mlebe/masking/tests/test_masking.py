# -*- coding: utf-8 -*-
import os
import tempfile
from pathlib import Path
from unittest import TestCase

import nibabel as nib
import numpy as np

from mlebe.masking.predict_mask import predict_mask


class TestMlebeMasking(TestCase):
    """Tests of the masking for the mlebe package."""

    def test_masking(self):
        """Test the masking function."""
        with tempfile.TemporaryDirectory() as test_dir:
            test_in_file = np.ones((63, 96, 48))
            test_in_file = nib.Nifti1Image(test_in_file, np.eye(4))
            test_in_file_path = Path(test_dir) / 'test_in_file.nii.gz'
            nib.save(test_in_file, test_in_file_path)
            for input_type in ['anat', 'func']:
                nii_path_masked, [resampled_mask_path], resampled_mask_path = predict_mask(in_file=test_in_file_path,
                                                                                           input_type=input_type)

                assert Path(
                    nii_path_masked).exists(), f'nii_path_masked "{nii_path_masked}" does not exist, but should exist.'
                assert Path(
                    resampled_mask_path).exists(), f'resampled_mask_path "{resampled_mask_path}" does not exist, ' \
                                                   f'but should exist.'
