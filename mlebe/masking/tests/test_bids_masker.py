# -*- coding: utf-8 -*-
"""Tests of the BidsMasker class."""

import json
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from mlebe.masking.bids_masker import BidsMasker
from mlebe.masking.predict_mask import predict_mask
from modun.file_io import dict2json

from mlebe.masking.tests.utils import create_toy_bids_directory


@pytest.mark.tox
def test_BidsMasker():
    """Test the BidsMasker function."""
    with tempfile.TemporaryDirectory() as test_dir:
        test_dir = Path(test_dir)

        create_toy_bids_directory(dest_dir=test_dir)

        config = {
            "masking_config": {
                "masking_config_anat": {
                    "test": True
                },
                "masking_config_func": {
                    "test": True,
                }
            }
        }
        config_path = test_dir / 'config.json'
        dict2json(d=config, out_path=config_path)

        bids_masker = BidsMasker(test_dir, str(config_path))

        bids_masker.run()


if __name__ == '__main__':
    test_BidsMasker()
