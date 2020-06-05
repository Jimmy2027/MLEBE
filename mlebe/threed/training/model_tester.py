import os

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataio.loaders import get_dataset
from dataio.transformation import get_dataset_transformation
from models import get_model
from utils.error_logger import StatLogger
from utils.metrics import dice_score, precision_and_recall
from utils.utils import json_file_to_pyobj, mkdir

save_directory = 'temp/'
mkdir(save_directory)
json_opts = json_file_to_pyobj('configs/load_pretrained_mlebe_config.json')
data_dir = json_opts.data.data_dir
model_path = 'checkpoints/test/120_net_S.pth'
template_dir = '/usr/share/mouse-brain-atlases/'

model = get_model(json_opts.model)
ds_class = get_dataset('mlebe_dataset')
ds_path = json_opts.data.data_dir
channels = json_opts.data_opts.channels
split_opts = json_opts.data_split
train_opts = json_opts.training
ds_transform = get_dataset_transformation('mlebe', opts=json_opts.augmentation,
                                          max_output_channels=json_opts.model.output_nc)

test_dataset = ds_class(template_dir, ds_path, json_opts.data.studies, split='test', save_dir=None,
                        data_type='anat', transform=ds_transform['valid'],
train_size = split_opts.train_size, test_size = split_opts.test_size,
                                                valid_size = split_opts.validation_size, split_seed = split_opts.seed)

test_loader = DataLoader(dataset=test_dataset, num_workers=16, batch_size=train_opts.batchSize, shuffle=False)

# Setup stats logger
stat_logger = StatLogger()

# test
for iteration, (images, labels, indices) in tqdm(enumerate(test_loader, 1), total=len(test_loader)):
    model.set_input(images, labels)
    model.test()
    ids = test_dataset.get_ids(indices)

    for batch_iter in range(len(ids)):
        input_arr = np.squeeze(images[batch_iter].cpu().numpy()).astype(np.float32)
        label_arr = np.squeeze(labels[batch_iter].cpu().numpy()).astype(np.int16)
        output_arr = np.squeeze(model.pred_seg.cpu().byte().numpy()).astype(np.int16)[batch_iter]

        dice_vals = dice_score(label_arr, output_arr, n_class=int(1))
        # md, hd = distance_metric(label_arr, output_arr, dx=2.00, k=1)
        precision, recall = precision_and_recall(label_arr, output_arr, n_class=int(1))
        stat_logger.update(split='test', input_dict={'img_name': ids[batch_iter],
                                                     'dice': dice_vals[0],
                                                     'precision': precision[0],
                                                     'reca_MYO': recall[0],
                                                     # 'md_MYO': md,
                                                     # 'hd_MYO': hd,
                                                     })
        # Write a nifti image
        import SimpleITK as sitk

        input_img = sitk.GetImageFromArray(np.transpose(input_arr, (2, 1, 0)));
        input_img.SetDirection([-1, 0, 0, 0, -1, 0, 0, 0, 1])
        label_img = sitk.GetImageFromArray(np.transpose(label_arr, (2, 1, 0)));
        label_img.SetDirection([-1, 0, 0, 0, -1, 0, 0, 0, 1])
        predi_img = sitk.GetImageFromArray(np.transpose(output_arr, (2, 1, 0)));
        predi_img.SetDirection([-1, 0, 0, 0, -1, 0, 0, 0, 1])

        sitk.WriteImage(input_img, os.path.join(save_directory, '{}_img.nii.gz'.format(ids[batch_iter])))
        sitk.WriteImage(label_img, os.path.join(save_directory, '{}_lbl.nii.gz'.format(ids[batch_iter])))
        sitk.WriteImage(predi_img, os.path.join(save_directory, '{}_pred.nii.gz'.format(ids[batch_iter])))

        stat_logger.statlogger2csv(split='test', out_csv_name=os.path.join(save_directory, 'stats.csv'))
        for key, (mean_val, std_val) in stat_logger.get_errors(split='test').items():
            print('-', key, ': \t{0:.3f}+-{1:.3f}'.format(mean_val, std_val), '-')
