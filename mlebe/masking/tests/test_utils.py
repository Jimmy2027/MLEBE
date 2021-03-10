# -*- coding: utf-8 -*-
from unittest import TestCase

from mlebe.masking.utils import get_mlebe_models


class TestMlebeUtilities(TestCase):
    """Tests for the utilities of the mlebe package."""

    def test_get_mlebe_models(self):
        """Test if the function get_mlebe_models downloads the pretrained models."""
        for input_type in ['anat', 'func']:
            model_folder_path = get_mlebe_models(input_type)
            assert model_folder_path.exists(), f'model_folder_path "{model_folder_path}" does not exist.'
