import copy
import os

import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from mlebe.training.dataio.loaders import get_dataset
from mlebe.training.dataio.loaders.utils import arrange_mask, load_mask
from mlebe.training.dataio.transformation import get_dataset_transformation
from mlebe.training.models import get_model
from mlebe.training.models.layers.loss import dice
from mlebe.training.utils.utils import json_file_to_pyobj, mkdir

study = ['irsabi_dargcc', 'irsabi']
slice_view = 'coronal'
IMG_NBRs = [65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65]


def evaluate(config_path):
    json_opts = json_file_to_pyobj(config_path)
    template_dir = json_opts.data.template_dir
    model = get_model(json_opts.model)
    save_path = os.path.join(model.save_dir, 'irsabi_test')
    mkdir(save_path)
    data_type = json_opts.data.data_type
    print(save_path)
    # shape of the images on which the classifier was trained:
    training_shape = json_opts.augmentation.mlebe.scale_size[:3]
    ds_class = get_dataset('mlebe_dataset')
    # define preprocessing transformations for model
    ds_transform = get_dataset_transformation('mlebe', opts=json_opts.augmentation,
                                              max_output_channels=json_opts.model.output_nc)

    test_dataset = ds_class(template_dir, json_opts.data.data_dir, json_opts.data, split='test',
                            transform=ds_transform['valid'],
                            train_size=None, training_shape=training_shape)
    data_selection = test_dataset.data_selection
    transformer = ds_transform['valid']()

    temp = load_mask(template_dir)
    mask_data = [copy.deepcopy(temp) for _ in range(len(data_selection))]
    dice_scores_df = pd.DataFrame(columns=['volume_name', 'slice', 'dice_score', 'idx'])
    predictions = []
    for volume in tqdm(range(len(data_selection))):  # volume is an index
        # get volume
        volume_name = data_selection.iloc[volume]['uid']
        img = nib.load(data_selection.iloc[volume]['path']).get_data()
        target = mask_data[volume].get_data()

        if json_opts.data.with_arranged_mask:
            # set the mask to zero where the image is zero
            target = arrange_mask(img, target)

        # img = preprocess(img, training_shape[:2], 'coronal')
        # target = preprocess(target, training_shape[:2], 'coronal')
        #
        # # set image shape to x,y,z
        # img = np.moveaxis(img, 0, 2)
        # target = np.moveaxis(target, 0, 2)

        # preprocess data for compatibility with model
        network_input = transformer(np.expand_dims(img, -1))
        target = np.squeeze(transformer(np.expand_dims(target, -1)).cpu().byte().numpy()).astype(np.int16)
        # add dimension for batches
        network_input = network_input.unsqueeze(0)
        model.set_input(network_input)
        model.test()
        # predict
        mask_pred = np.squeeze(model.pred_seg.cpu().numpy())
        img = np.squeeze(network_input.numpy())
        # set image shape to z,x,y
        mask_pred = np.moveaxis(mask_pred, 2, 0)
        img = np.moveaxis(img, 2, 0)
        target = np.moveaxis(target, 2, 0)

        for slice in range(img.shape[0]):
            dice_score = dice(target[slice], mask_pred[slice])
            # see if this is a black slice (want to skip those for visualisation)
            black_slice = np.max(img[slice]) <= 0
            dice_scores_df = dice_scores_df.append(
                {'volume_name': volume_name, 'slice': slice, 'dice_score': dice_score, 'idx': volume,
                 'black_slice': black_slice},
                ignore_index=True)
        predictions.append(mask_pred)
    min_df = dice_scores_df.loc[dice_scores_df['black_slice'] == False].sort_values(by=['dice_score']).head(
        sum(IMG_NBRs) // 2)
    min_df = pd.concat([min_df,
                        dice_scores_df.loc[dice_scores_df['black_slice'] == False].sort_values(by=['dice_score']).tail(
                            sum(IMG_NBRs) - sum(IMG_NBRs) // 2)],
                       ignore_index=True)
    with PdfPages(os.path.join(save_path, 'irsabi_test_{}.pdf'.format(data_type))) as pdf:
        df_idx = 0

        for IMG_NBR in IMG_NBRs:
            plt.figure(figsize=(40, IMG_NBR * 10))
            plt.figtext(.5, .9, 'Mean dice score of {}'.format(np.round(dice_scores_df['dice_score'].mean(), 4)),
                        fontsize=100, ha='center')
            i = 1
            while i <= IMG_NBR * 2 and df_idx < len(min_df):
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


if __name__ == '__main__':
    evaluate(
        '/home/hendrik/src/MLEBE/mlebe/training/three_D/checkpoints/2020_7_14_1_35_43/trained_mlebe_config_anat.json')
