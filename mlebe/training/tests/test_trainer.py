import json
import os
import tempfile

from mlebe.training.train_segmentation import train


def test_train():
    with tempfile.TemporaryDirectory() as test_dir:
        test_config = {
            "training": {
                "arch_type": "gsd_pCT",
                "n_epochs": 1,
                "save_epoch_freq": 10,
                "lr_policy": "step",
                "lr_decay_iters": 250,
                "batchSize": 2,
                "preloadData": False,
                "early_stopping_patience": 10
            },
            "visualisation": {
                "display_port": 8097,
                "no_html": True,
                "save_epoch_freq": 10,
                "display_winsize": 256,
                "display_id": -1,
                "display_single_pane_ncols": 0
            },
            "data_split": {
                "train_size": 0.7,
                "test_size": 0.15,
                "validation_size": 0.15,
                "seed": 42
            },
            "data": {
                "studies": [
                    "mgtdbs"
                ],
                "excluded_from_training": [
                    "irsabi"
                ],
                "slice_view": "coronal",
                "data_dir": "/mnt/data/hendrik/mlebe_data/",
                "template_dir": "/usr/share/mouse-brain-atlases/",
                "data_type": "anat",
                "with_arranged_mask": False,
                "with_blacklist": True,
                "blacklist_dir": "~/docsrc/mlebe/data/Blacklist",
                "func_training_dir": "~/var/tmp/func_training"
            },
            "data_opts": {
                "channels": [
                    1
                ]
            },
            "augmentation": {
                "mlebe": {
                    "shift": [
                        0.5,
                        0.5
                    ],
                    "rotate": 25,
                    "scale_val": [
                        0.7,
                        1.3
                    ],
                    "max_deform": [
                        5,
                        5,
                        5
                    ],
                    "intensity": [
                        0.5,
                        1.5
                    ],
                    "random_flip_prob": 0,
                    "random_affine_prob": 0,
                    "random_elastic_prob": 0,
                    "scale_range": [0.7, 1.2],
                    "scale_proba": 1,
                    "scale_size": [
                        64,
                        64,
                        96,
                        1
                    ],
                    "bias_magnitude_range": 1.0,
                    "random_noise_prob": 0.7,
                    "normalization": 'mlebe',
                }
            },
            "model": {
                "type": "seg",
                "continue_train": False,
                "which_epoch": -1,
                "model_type": "unet_pct_multi_att_dsv",
                "tensor_dim": "3D",
                "division_factor": 16,
                "input_nc": 1,
                "output_nc": 2,
                "lr_rate": 0.0001,
                "l2_reg_weight": 1e-06,
                "feature_scale": 4,
                "gpu_ids": [
                    0
                ],
                "isTrain": True,
                "checkpoints_dir": test_dir,
                "experiment_name": "test",
                "criterion": "focal_tversky_loss"
            }
        }
        with open(os.path.join(test_dir, 'test_config.json'), 'w') as jsonfile:
            json.dump(test_config, jsonfile, indent=4)
        _ = train(os.path.join(test_dir, 'test_config.json'))


if __name__ == '__main__':
    test_train()
