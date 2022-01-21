# -*- coding: utf-8 -*-

import os
import shutil
from pathlib import Path
from typing import Mapping, Optional, List
from typing import Tuple

import nibabel as nib
import numpy as np
from nipype.interfaces.fsl.maths import MeanImage
from norby.utils import norby
from samri.pipelines.utils import bids_data_selection
from tqdm import tqdm

from mlebe import log
from mlebe.masking.utils import get_mask, get_mlebe_models, get_biascorrect_opts_defaults, save_masking_visualisation, \
    crop_bids_image
from mlebe.masking.utils import remove_outliers, get_masking_opts, reconstruct_image, pad_to_shape, \
    get_model_config
from mlebe.training.models import get_model


class BidsMasker:
    """
    Mask the brain region of scans in a bids format.
    """

    def __init__(self, work_dir: Path, masking_config_path: str, structural_match: Optional[Mapping[str, list]] = False,
                 functional_match: Optional[Mapping[str, list]] = False, subjects: List[str] = False,
                 sessions: List[str] = False):
        """

        Args:
            work_dir (Path): Path to directory containing a folder with the bids images.
        """
        self.masking_config_path = masking_config_path
        self.work_dir = work_dir
        self.bids_path = work_dir / 'bids'
        self.preprocessing_dir = work_dir / 'preprocessing'
        self.data_selection = bids_data_selection(base=str(self.bids_path), structural_match=structural_match,
                                                  functional_match=functional_match, subjects=subjects,
                                                  sessions=sessions)

        self.tmean_dir = self.preprocessing_dir / 'bids_tmean'
        self.tmean_dir.mkdir(exist_ok=True, parents=True)

    def run(self):
        self.run_masking('anat')
        self.get_f_means()
        self.run_masking('func')

        self.data_selection.to_csv(self.preprocessing_dir / 'masked_bids' / 'data_selection.csv')

        # move log file to preprocessing dir
        log_file = Path(log.manager.root.handlers[1].baseFilename)
        shutil.move(log_file, self.preprocessing_dir / 'masked_bids' / log_file.name)

    def get_f_means(self):
        data_selection_func = self.data_selection.loc[self.data_selection.datatype == 'func']
        for _, elem in tqdm(data_selection_func.iterrows(), total=len(data_selection_func), postfix='Tmean'):
            in_path = elem.path
            out_dir = self.tmean_dir / f'sub-{elem.subject}' / f'ses-{elem.session}' / elem.datatype
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / elem.path.split('/')[-1]
            mean_image = MeanImage(in_file=in_path, dimension='T', out_file=out_path)
            mean_image.run()

            self.data_selection.loc[self.data_selection.path == elem.path, 'tmean_path'] = out_path

    def run_masking(self, data_type: str) -> None:
        """
        Mask all bids scans in self.data_selection with data type "data_type".

        Parameters
        ----------
        data_type : either anat or func
        """
        masking_opts = get_masking_opts(self.masking_config_path, data_type)

        if 'model_folder_path' not in masking_opts or not masking_opts['model_folder_path']:
            # if no model_folder_path is given in the config, the default models are selected.
            masking_opts['model_folder_path'] = get_mlebe_models(data_type)
        model_config = get_model_config(masking_opts)

        # load model
        model = get_model(model_config.model)
        df = self.data_selection.loc[self.data_selection.datatype == data_type]
        for _, elem in tqdm(df.iterrows(), total=len(df)):
            bids_path = str(elem.path if data_type == 'anat' else elem.tmean_path)
            mask_path, masked_path = self.mask_one(bids_path, elem, masking_opts, model, model_config)

            self.data_selection.loc[self.data_selection.path == elem.path, 'mask_path'] = mask_path
            self.data_selection.loc[self.data_selection.path == elem.path, 'masked_path'] = masked_path

    def mask_one(self, bids_path: str, elem, masking_opts, model, model_config) -> Tuple[Path, Path]:
        bids_file = nib.load(bids_path)
        bids_file_data = bids_file.get_data()

        resampled_path = 'resampled_input.nii.gz'
        resampled_nii_path = os.path.abspath(os.path.expanduser(resampled_path))

        # resample bids image into the voxel space for which the masking model was trained.
        resample_cmd = f'ResampleImage 3 {bids_path}  {resampled_nii_path} 0.2x0.2x0.2'
        os.system(resample_cmd)
        log.info(f'Resample image with "{resample_cmd}"')

        # crop bids image one the side with given values to alleviate the task of the masking model.
        if 'crop_values' in masking_opts and masking_opts['crop_values']:
            crop_bids_image(resampled_nii_path, masking_opts['crop_values'])

        if 'bias_field_correction' in masking_opts and masking_opts['bias_field_correction']:
            bias_correction_config = get_biascorrect_opts_defaults(masking_opts)
            bias_corrected_path = os.path.abspath(os.path.expanduser('corrected_input.nii.gz'))

            command = 'N4BiasFieldCorrection --bspline-fitting {} -d 3 --input-image {} --convergence {} --output {} --shrink-factor {}'.format(
                bias_correction_config['bspline_fitting'], resampled_nii_path,
                bias_correction_config['convergence'],
                bias_corrected_path, bias_correction_config['shrink_factor'])

            os.system(command)
            log.info(f'Apply bias correction with "{command}"')

        else:
            bias_corrected_path = resampled_nii_path

        bias_corrected_img = nib.load(bias_corrected_path)
        bias_corrected_img_data = bias_corrected_img.get_data()

        # get the mask
        resampled_shape = np.moveaxis(bias_corrected_img_data, 2, 0).shape
        in_file_data, mask_pred, model_input = get_mask(model_config, bias_corrected_img_data, resampled_shape,
                                                        use_cuda=masking_opts['use_cuda'], model=model)
        mask_pred = remove_outliers(mask_pred)

        # resample the predicted mask
        resampled_mask_path = self.reconstruct_to_imgsize(bids_img=bids_file, bias_corrected_img=bias_corrected_img,
                                                          ori_shape=resampled_shape, mask_pred=mask_pred)
        resampled_mask = nib.load(resampled_mask_path)
        resampled_mask_data = resampled_mask.get_data()

        if resampled_mask_data.shape != bids_file_data.shape:
            resampled_mask_data = pad_to_shape(resampled_mask_data, bids_file_data)

        nib.save(nib.Nifti1Image(resampled_mask_data, bids_file.affine, bids_file.header),
                 resampled_mask_path)

        # Masking of the input image
        masked_image_data = np.multiply(resampled_mask_data, bids_file_data) \
            .astype('float32')  # nibabel gives a non-helpful error if trying to save data that has dtype float64

        masked_image = nib.Nifti1Image(masked_image_data, bids_file.affine, bids_file.header)

        if 'visualisation_path' in masking_opts and masking_opts['visualisation_path']:
            log.info(f'visualisation_path is {masking_opts["visualisation_path"]}')
            save_masking_visualisation(masking_opts, Path(bids_path).name, model_input=model_input,
                                       predicted_mask=mask_pred, input_data=np.moveaxis(bids_file_data, 2, 0),
                                       resampled_mask=np.moveaxis(masked_image_data, 2, 0))

        # saving results
        filename = bids_path.split('/')[-1]
        out_dir = self.preprocessing_dir / 'masked_bids' / f'sub-{elem.subject}' / f'ses-{elem.session}' / elem.datatype
        out_dir.mkdir(parents=True, exist_ok=True)
        mask_path = out_dir / f'mask_{filename}'
        masked_path = out_dir / f'masked_{filename}'
        nib.save(nib.Nifti1Image(resampled_mask_data, bids_file.affine, bids_file.header), mask_path)
        nib.save(masked_image, masked_path)
        return mask_path, masked_path

    @staticmethod
    def reconstruct_to_imgsize(bids_img, bias_corrected_img, ori_shape: Tuple[int], mask_pred: np.ndarray) -> str:
        """
        Reconstruct the predicted mask to the original bids image size.
        First resize the mask to the resampled bids shape, then resample it to into the original voxel space.
        """
        resized = reconstruct_image(ori_shape, mask_pred)
        resized_path = 'resized_mask.nii.gz'
        resized_path = os.path.abspath(os.path.expanduser(resized_path))
        resized_mask = nib.Nifti1Image(resized, bias_corrected_img.affine, bias_corrected_img.header)
        nib.save(resized_mask, resized_path)

        # get voxel sizes from input
        input_img_affine = bids_img.affine
        voxel_sizes = nib.affines.voxel_sizes(input_img_affine)

        resampled_mask_path = 'resampled_mask.nii.gz'
        resampled_mask_path = os.path.abspath(os.path.expanduser(resampled_mask_path))
        resample_cmd = 'ResampleImage 3 {input} '.format(
            input=resized_path) + ' ' + resampled_mask_path + ' {x}x{y}x{z} '.format(x=voxel_sizes[0], y=voxel_sizes[1],
                                                                                     z=voxel_sizes[2]) + ' 0 1'
        log.info(f'Resample image with "{resample_cmd}"')
        os.system(resample_cmd)
        return resampled_mask_path


if __name__ == '__main__':
    with norby(whichbot='mlebe'):
        work_dir = Path('~/.scratch/mlebe').expanduser()
        # work_dir = Path('/home/hendrik/temp').expanduser()
        bids_masker = BidsMasker(work_dir, str(work_dir / 'config.json'))

        bids_masker.run()
