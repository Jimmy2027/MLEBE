import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from mlebe.threed.training.dataio.loaders import get_dataset
from mlebe.threed.training.dataio.transformation import get_dataset_transformation
from mlebe.threed.training.models import get_model
from mlebe.threed.training.utils.error_logger import StatLogger
from mlebe.threed.training.utils.metrics import distance_metric_wrapper, precision_and_recall, single_class_dice_score
from mlebe.threed.training.utils.utils import json_file_to_pyobj, rm_and_mkdir
from mlebe.training.utils.general import remove_black_images
from matplotlib import pyplot as plt


def tester(json_opts, test_dataset, save_directory):
    model = get_model(json_opts.model)
    train_opts = json_opts.training

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

            dice_val = single_class_dice_score(label_arr, output_arr)
            md, hd = distance_metric_wrapper(label_arr, output_arr, dx=2.00, k=1)
            precision, recall = precision_and_recall(label_arr, output_arr, n_class=int(1))
            stat_logger.update(split='test', input_dict={'img_name': ids[batch_iter],
                                                         'dice': dice_val,
                                                         'precision': precision[0],
                                                         'reca_MYO': recall[0],
                                                         'md': md,
                                                         'hd': hd,
                                                         })

            save_dir = os.path.join(save_directory, ids[batch_iter])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            input_img, target = remove_black_images(input_arr, label_arr)
            _, output_img = remove_black_images(input_arr, output_arr)

            y = input_img.shape[0]
            i = 1
            plt.figure(figsize=(30, y * 5))
            for slice in range(y):
                plt.subplot(y, 2, i)
                plt.imshow(input_img[slice], cmap='gray')
                plt.imshow(target[slice], cmap='Blues', alpha=0.6)
                plt.axis('off')
                i += 1
                plt.subplot(y, 2, i)
                plt.title(str(dice_val))
                plt.imshow(input_img[slice], cmap='gray')
                plt.imshow(output_img[slice], cmap='Blues', alpha=0.6)
                plt.axis('off')
                i += 1

            plt.savefig(save_dir + '/{}.{}'.format(ids[batch_iter], 'pdf'), format='pdf')
            plt.close()

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


if __name__ == '__main__':
    save_directory = 'temp/'
    rm_and_mkdir(save_directory)
    json_opts = json_file_to_pyobj('/home/hendrik/src/MLEBE/mlebe/threed/training/checkpoints/with_augmentations_func/trained_mlebe_config_func.json')
    data_dir = json_opts.data.data_dir
    template_dir = '/usr/share/mouse-brain-atlases/'

    ds_class = get_dataset('mlebe_dataset')
    ds_path = json_opts.data.data_dir
    channels = json_opts.data_opts.channels
    split_opts = json_opts.data_split
    train_opts = json_opts.training
    ds_transform = get_dataset_transformation('mlebe', opts=json_opts.augmentation,
                                              max_output_channels=json_opts.model.output_nc)

    test_dataset = ds_class(template_dir, ds_path, ['irsabi_dargcc', 'irsabi'], split='test', save_dir=None,
                            data_type='func', transform=ds_transform['valid'],
                            train_size=None)

    tester(json_opts, test_dataset, save_directory)
