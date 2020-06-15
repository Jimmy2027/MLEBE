import mlebe.training.utils.data_loader as dl
import mlebe.training.utils.general as utils
import mlebe.training.utils.scoring_utils as su
import copy
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mlebe.threed.training.models import get_model
from mlebe.threed.training.utils.utils import json_file_to_pyobj, mkdir
from mlebe.threed.training.dataio.transformation import get_dataset_transformation
from mlebe.threed.training.dataio.loaders import get_dataset
from mlebe.training.utils.general import preprocess, arrange_mask, remove_black_images
import nibabel as nib

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

template_dir = '/usr/share/mouse-brain-atlases/'
study = ['irsabi_dargcc', 'irsabi']
slice_view = 'coronal'
shape = (128, 128)
IMG_NBRs = [65, 65, 65, 65, 65, 65, 65, 65]


def evaluate(config_path):
    json_opts = json_file_to_pyobj(config_path)

    model = get_model(json_opts.model)
    save_path = os.path.join(model.save_dir, 'irsabi_test')
    mkdir(save_path)
    data_type = json_opts.data.data_type
    print(save_path)

    ds_class = get_dataset('mlebe_dataset')
    # define preprocessing transfromer for model
    ds_transform = get_dataset_transformation('mlebe', opts=json_opts.augmentation,
                                              max_output_channels=json_opts.model.output_nc)

    test_dataset = ds_class(template_dir, json_opts.data.data_dir, json_opts.data, split='test',
                            transform=ds_transform['valid'],
                            train_size=None)
    data_selection = test_dataset.data_selection
    transformer = ds_transform['valid']()

    mask_data = []
    temp = dl.load_mask(template_dir)
    for i in range(len(data_selection)):
        mask_data.append(copy.deepcopy(temp))

    dice_scores_df = pd.DataFrame(columns=['volume_name', 'slice', 'dice_score', 'idx'])
    predictions = []
    for volume in range(len(data_selection)):  # volume is an index
        # get volume
        volume_name = data_selection.iloc[volume]['uid']
        img = nib.load(data_selection.iloc[volume]['path']).get_data()
        target = mask_data[volume].get_data()

        # set image shape to x,y,z
        img = np.moveaxis(img, 2, 1)
        target = np.moveaxis(target, 2, 1)

        # img = preprocess(img, (128, 128), 'coronal')
        # target = preprocess(target, (128, 128), 'coronal')

        # preprocess data for compatibility with model
        network_input = transformer(np.expand_dims(img, -1))
        target = np.squeeze(transformer(np.expand_dims(target, -1)).cpu().byte().numpy()).astype(np.int16)
        # add dimension for batches
        network_input = network_input.unsqueeze(0)
        model.set_input(network_input)
        model.test()
        # predict
        mask_pred = np.squeeze(model.pred_seg.cpu().byte().numpy()).astype(np.int16)
        img = np.squeeze(network_input.cpu().byte().numpy()).astype(np.int16)
        # set image shape to z,x,y
        mask_pred = np.moveaxis(mask_pred, 2, 0)
        img = np.moveaxis(img, 2, 0)
        target = np.moveaxis(target, 2, 0)

        for slice in range(img.shape[0]):
            dice_score = su.dice(target[slice], mask_pred[slice])
            dice_scores_df = dice_scores_df.append(
                {'volume_name': volume_name, 'slice': slice, 'dice_score': dice_score, 'idx': volume},
                ignore_index=True)
        predictions.append(mask_pred)

    min_df = dice_scores_df.sort_values(by=['dice_score']).head(sum(IMG_NBRs))
    df_idx = 0

    with PdfPages(os.path.join(save_path,  'irsabi_test_{}.pdf'.format(data_type))) as pdf:
        for IMG_NBR in IMG_NBRs:
            plt.figure(figsize=(40, IMG_NBR * 10))
            plt.figtext(.5, .9, 'Mean dice score of {}'.format(np.round(dice_scores_df['dice_score'].mean(), 4)),
                        fontsize=100, ha='center')
            i = 1
            while i <= IMG_NBR * 2:
                volume = min_df.iloc[df_idx]['idx']
                slice = min_df.iloc[df_idx]['slice']
                dice_score = min_df.iloc[df_idx]['dice_score']
                plt.subplot(IMG_NBR, 2, i)
                plt.imshow(img[slice], cmap='gray')
                plt.imshow(target[slice], cmap='Blues', alpha=0.6)
                plt.axis('off')
                i += 1
                plt.subplot(IMG_NBR, 2, i)
                plt.imshow(img[slice], cmap='gray')
                plt.imshow(predictions[volume][slice], cmap='Blues', alpha=0.6)
                plt.title('Volume: {}, slice {}, dice {}'.format(volume_name, slice, dice_score))
                plt.axis('off')
                i += 1
                df_idx += 1
            pdf.savefig()
            plt.close()

    plt.title('Dice score = {}'.format(dice_scores_df['dice_score'].mean()))
    plt.savefig('{}.pdf'.format(save_path), format='pdf')


    return dice_scores_df['dice_score'].mean(), dice_scores_df['dice_score'].std()

