import nibabel as nib
from data_augment import augment
from matplotlib import pyplot as plt
import numpy as np
import tensorflow.keras.preprocessing as kp

x = nib.load('/home/hendrik/.scratch/drlfom/dw_preprocessing/sub-5668/ses-ofM/anat/sub-5668_ses-ofM_acq-TurboRARE_T2w.nii.gz')
# x = nib.load('/usr/share/mouse-brain-atlases/dsurqec_200micron_mask.nii')

x = x.get_data()
x = np.moveaxis(x, 1, 0)

image = np.expand_dims(x, -1)
# image = image[20]
# image = np.expand_dims(image, 0)

for i in range(np.shape(x)[0]):
    plt.figure()
    plt.subplot(2, 4, 1)
    plt.imshow(np.squeeze(augment(image[i], brightness_range = (1, 1), noise_var_range = (0,0))), cmap = 'gray')
    plt.axis('off')
    plt.subplot(2, 4, 2)
    plt.imshow(np.squeeze(augment(image[i], brightness_range = (1, 1), noise_var_range = (0.01, 0.01))), cmap = 'gray')
    plt.axis('off')
    plt.subplot(2, 4, 3)
    plt.imshow(np.squeeze(augment(image[i], brightness_range = (1, 1), noise_var_range = (0.02, 0.02))), cmap = 'gray')
    plt.axis('off')
    plt.subplot(2, 4, 4)
    plt.imshow(np.squeeze(augment(image[i], brightness_range = (1, 1), noise_var_range = (0.1, 0.1))), cmap = 'gray')
    plt.axis('off')
    plt.subplot(2, 4, 5)
    plt.imshow(np.squeeze(augment(image[i], brightness_range = (0.5, 0.5), noise_var_range = (0, 0.01))), cmap = 'gray')
    plt.title('this')
    plt.axis('off')
    plt.subplot(2, 4, 6)
    plt.imshow(np.squeeze(augment(image[i], brightness_range = (0.5, 0.5), noise_var_range = (0,0))), cmap = 'gray')
    plt.axis('off')
    plt.subplot(2, 4, 7)
    plt.imshow(np.squeeze(augment(image[i], brightness_range = (0.4, 0.4), noise_var_range = (0, 0.01))), cmap = 'gray')
    plt.axis('off')
    plt.subplot(2, 4, 8)
    plt.imshow(np.squeeze(augment(image[i], brightness_range = (0.4, 0.4), noise_var_range = (0,0))), cmap = 'gray')
    plt.axis('off')
    plt.savefig('vis/{}'.format(i))
    plt.close()



# data_gen_args = dict(rotation_range=90,
#                      width_shift_range=30,
#                      height_shift_range=30,
#                      shear_range=5,
#                      zoom_range=0.3,
#                      horizontal_flip=True,
#                      vertical_flip=True,
#                      fill_mode='nearest')
# image_datagen = kp.image.ImageDataGenerator(**data_gen_args)
# image_generator = image_datagen.flow(image, seed=1)
# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(np.squeeze(image))
# plt.axis('off')
# plt.subplot(1,2,2)
# plt.imshow(np.squeeze(image_generator.next()[0]))
# plt.axis('off')
# plt.savefig('vis/keras_example')
