import nibabel as nib
from data_augment import augment
from matplotlib import pyplot as plt
import numpy as np
import tensorflow.keras.preprocessing as kp
from mlebe.training.utils import preprocess

data_gen_args = dict(
    rotation_range=90,
    width_shift_range=30,
    height_shift_range=30,
    shear_range=2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    brightness_range = (0.7, 1.3),
    noise_var_range = (0, 0.001),    #variance range of the gaussian noise that is added to the image
    bias_var_range = (10, 20),   #width of the bias added on top of the images
    )

x = nib.load('/home/hendrik/.scratch/drlfom/dw_preprocessing/sub-5668/ses-ofM/anat/sub-5668_ses-ofM_acq-TurboRARE_T2w.nii.gz')
mask = nib.load('/usr/share/mouse-brain-atlases/dsurqec_200micron_mask.nii')
# x = nib.load('/usr/share/mouse-brain-atlases/dsurqec_200micron_mask.nii')
shape = (128, 128)
x = preprocess(x.get_data(), shape, 'coronal')
mask = preprocess(mask.get_data(), shape, 'coronal')
mask = np.expand_dims(mask, -1)
image = np.expand_dims(x, -1)

mask_datagen = image_datagen = kp.image.ImageDataGenerator({key: data_gen_args[key] for key in data_gen_args.keys() if not key in ['brightness_range', 'noise_var_range', 'bias_var_range']})
image_generator = image_datagen.flow(image, seed=1)
mask_generator = image_datagen.flow(mask, seed=1)
imgs = [next(image_generator) for _ in range(10)]
masks = [np.where(next(mask_generator) > 0.5, 1, 0).astype('float32') for _ in range(10)]
imgs = np.concatenate(imgs)
masks = np.concatenate(masks)
for i in range(len(imgs)):
    img = np.squeeze(augment(imgs[i], masks[i], brightness_range = data_gen_args['brightness_range'], noise_var_range = data_gen_args['noise_var_range'], bias_var_range = data_gen_args['bias_var_range']))
    plt.imshow(img, cmap = 'gray')
    plt.axis('off')
    plt.savefig('vis/{}'.format(i))
    plt.close()

