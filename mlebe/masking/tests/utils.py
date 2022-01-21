from pathlib import Path
from typing import Iterable, Optional
import nibabel as nib

import numpy as np


def create_toy_bids_directory(dest_dir: Path, subjects: Optional[Iterable[str]] = None,
                              types: Optional[Iterable[str]] = None):
    if subjects is None:
        subjects = ["sub-4001"]
    if types is None:
        types = ['ses-ofM']

    bids_dir = dest_dir / 'bids'
    bids_dir.mkdir()

    for subject in subjects:
        for type in types:
            for data_type in ['anat', 'func']:
                subject_dir = bids_dir / subject / type / data_type
                subject_dir.mkdir(parents=True)

                subject_data = np.ones((63, 96, 48))
                subject_file = nib.Nifti1Image(subject_data, np.eye(4))

                if data_type == 'anat':
                    subject_fn = subject_dir / f'{subject}_{type}_acq-TurboRARElowcov_T2w.nii.gz'
                else:
                    subject_fn = subject_dir / f'{subject}_{type}_task-JogB_acq-EPIlowcov_run-0_bold.nii.gz'

                nib.save(subject_file, subject_fn)
