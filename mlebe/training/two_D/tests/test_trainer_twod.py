import os
import shutil

import nibabel as nib
import numpy as np

import mlebe
from mlebe.training.two_D.trainer_config import trainer_config


# todo use tempfile?
# todo make this faster (remove all visualisations in test-mode)

def test_trainer():
    test_dir = os.path.join(os.path.dirname(mlebe.__file__), 'training/two_D/tests/temp/')
    os.makedirs(test_dir + 'preprocessing_test/', exist_ok=True)
    test_mask = np.ones((63, 96, 48))
    test_mask = nib.Nifti1Image(test_mask, np.eye(4))
    nib.save(test_mask, test_dir + 'dsurqec_200micron_mask.nii')
    test_scan = np.ones((63, 96, 48))
    test_scan = nib.Nifti1Image(test_scan, np.eye(4))
    nib.save(test_scan, test_dir + 'preprocessing_test/test_scan1_T2w.nii.gz')
    nib.save(test_scan, test_dir + 'preprocessing_test/test_scan2_T2w.nii.gz')
    anat_tester = trainer_config(file_name='test',
                                 data_dir=os.path.join(os.path.dirname(mlebe.__file__), 'training/tests/'),
                                 template_dir=os.path.join(os.path.dirname(mlebe.__file__), 'training/tests/temp/'),
                                 slice_view='coronal',
                                 shape=(128, 128), loss='dice', epochss=[1],
                                 data_gen_argss=[None],
                                 blacklist=False, data_sets=['temp'],
                                 test=True)
    anat_tester.train()
    shutil.rmtree(test_dir)
    shutil.rmtree(os.path.join(os.path.dirname(mlebe.__file__), 'training/tests/results'))


if __name__ == '__main__':
    test_trainer()
