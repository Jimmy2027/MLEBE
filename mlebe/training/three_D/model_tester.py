import os

import numpy as np
from mlebe.training.three_D.dataio.loaders import get_dataset
from mlebe.training.three_D.dataio.transformation import get_dataset_transformation
from mlebe.training.three_D.models import get_model
from mlebe.training.three_D.utils.utils import json_file_to_pyobj, rm_and_mkdir
from torch.utils.data import DataLoader
from tqdm import tqdm

from mlebe.training.two_D.utils.general import remove_black_images


def tester(json_opts, test_dataset, save_directory):
    model = get_model(json_opts.model)
    train_opts = json_opts.training

    test_loader = DataLoader(dataset=test_dataset, num_workers=16, batch_size=train_opts.batchSize, shuffle=False)

    # test
    x_test = []
    y_test = []
    y_pred = []
    for iteration, (images, labels, indices) in tqdm(enumerate(test_loader, 1), total=len(test_loader)):
        model.set_input(images, labels)
        model.test()
        ids = test_dataset.get_ids(indices)

        for batch_iter in range(len(ids)):
            input_arr = np.squeeze(images[batch_iter].cpu().numpy()).astype(np.float32)
            label_arr = np.squeeze(labels[batch_iter].cpu().numpy()).astype(np.int16)
            output_arr = np.squeeze(model.pred_seg.cpu().byte().numpy()).astype(np.int16)[batch_iter]

            input_img, target = remove_black_images(input_arr, label_arr)
            _, output_img = remove_black_images(input_arr, output_arr)

            y = input_img.shape[2]
            for slice in range(y):
                x_test.append(input_img[..., slice])
                y_test.append(target[..., slice])
                y_pred.append(output_img[..., slice])

    with open(os.path.join(save_directory, 'x_test.npy'), 'wb') as file1:
        np.save(file1, x_test)
    with open(os.path.join(save_directory, 'y_test.npy'), 'wb') as file2:
        np.save(file2, y_test)
    with open(os.path.join(save_directory, 'y_pred.npy'), 'wb') as file3:
        np.save(file3, y_pred)


if __name__ == '__main__':
    save_directory = 'temp/'
    rm_and_mkdir(save_directory)
    json_opts = json_file_to_pyobj(
        '/home/hendrik/src/MLEBE/mlebe/three_D/training/checkpoints/bias_field_prob-0.5_criterion-dice_loss_model_type-unet_pct_multi_att_dsv_normalization-normalize_medic_scale_range-[0.7, 1.2]_scale_size-[128, 128, 96, 1]_with_arranged_mask-True_with_blacklist-False2/trained_mlebe_config_func.json')
    data_dir = json_opts.data.data_dir
    template_dir = '/usr/share/mouse-brain-atlases/'

    ds_class = get_dataset('mlebe_dataset')
    ds_path = json_opts.data.data_dir
    channels = json_opts.data_opts.channels
    split_opts = json_opts.data_split
    train_opts = json_opts.training
    ds_transform = get_dataset_transformation('mlebe', opts=json_opts.augmentation,
                                              max_output_channels=json_opts.model.output_nc)

    test_dataset = ds_class(template_dir, ds_path, json_opts.data, split='test', save_dir=None,
                            transform=ds_transform['valid'],
                            train_size=split_opts.train_size, test_size=split_opts.test_size,
                            valid_size=split_opts.validation_size, split_seed=split_opts.seed,
                            training_shape=json_opts.augmentation.mlebe.scale_size[:3])

    tester(json_opts, test_dataset, save_directory)
