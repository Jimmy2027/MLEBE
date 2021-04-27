# -*- coding: utf-8 -*-
"""Tests of the masking for the mlebe package."""
import json
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from mlebe.masking.predict_mask import predict_mask


def test_masking_default():
    """Test the masking function."""
    with tempfile.TemporaryDirectory() as test_dir:
        test_in_file = np.ones((63, 96, 48))
        test_in_file = nib.Nifti1Image(test_in_file, np.eye(4))
        test_in_file_path = Path(test_dir) / 'test_in_file.nii.gz'
        nib.save(test_in_file, test_in_file_path)
        for input_type in ['func', 'anat']:
            config = {"masking_config": {f"masking_config_{input_type}": {"testing": True}}}
            # config = {"masking_config": {f"masking_config_{input_type}": {"testing": False}}}
            config_path = Path(test_dir) / 'test_config.json'
            with open(config_path, 'w') as jsonfile:
                json.dump(config, jsonfile, indent=4)

            nii_path_masked, [resampled_mask_path], resampled_mask_path = predict_mask(in_file=test_in_file_path,
                                                                                       input_type=input_type,
                                                                                       masking_config_path=config_path)

            assert Path(
                nii_path_masked).exists(), f'nii_path_masked "{nii_path_masked}" does not exist, but should exist.'
            assert Path(
                resampled_mask_path).exists(), f'resampled_mask_path "{resampled_mask_path}" does not exist, ' \
                                               f'but should exist.'


def test_masking_biascorrection():
    """Test the masking function using bias correction."""
    with tempfile.TemporaryDirectory() as test_dir:
        test_in_file = np.ones((63, 96, 48))
        test_in_file = nib.Nifti1Image(test_in_file, np.eye(4))
        test_in_file_path = Path(test_dir) / 'test_in_file.nii.gz'
        nib.save(test_in_file, test_in_file_path)
        for input_type in ['anat', 'func']:
            config = {"masking_config": {
                F"masking_config_{input_type}": {"bias_field_correction": {"shrink_factor": 2}, "testing": True}}}
            config_path = Path(test_dir) / 'test_config.json'

            with open(config_path, 'w') as jsonfile:
                json.dump(config, jsonfile, indent=4)
            nii_path_masked, [resampled_mask_path], resampled_mask_path = predict_mask(in_file=test_in_file_path,
                                                                                       input_type=input_type,
                                                                                       masking_config_path=config_path)

            assert Path(
                nii_path_masked).exists(), f'nii_path_masked "{nii_path_masked}" does not exist, but should exist.'
            assert Path(
                resampled_mask_path).exists(), f'resampled_mask_path "{resampled_mask_path}" does not exist, ' \
                                               f'but should exist.'


if __name__ == '__main__':
    test_masking_default()
    test_masking_biascorrection()
