from mlebe.masking.predict_mask import predict_mask
import mlebe
import os
import numpy as np
import nibabel as nib
import shutil

def test_masker():
    test_dir = os.path.join(os.path.dirname(mlebe.__file__), 'training/tests/temp/')
    os.makedirs(test_dir, exist_ok= True)
    test_in_file = np.ones((63, 96, 48))
    test_in_file = nib.Nifti1Image(test_in_file, np.eye(4))
    test_in_file_dir = os.path.join(test_dir, 'test_in_file.nii.gz')
    nib.save(test_in_file, test_in_file_dir)
    _,_,_ = predict_mask(test_in_file_dir, '', test= True)
    shutil.rmtree(test_dir)
