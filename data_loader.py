import os
import nibabel as nib
from matplotlib import pyplot as plt

data_dir = '~/.scratch/drlfom/dw_preprocessing'
data_dir = os.path.abspath(os.path.expanduser(data_dir))

im_data = []

def visu1():
    for o in os.listdir(data_dir):
        if not o.startswith('.') and os.path.isdir(os.path.join(data_dir,o)):
            for x in os.listdir(os.path.join(data_dir,o)):
                if x == 'ses-ofM':
                    for y in os.listdir(os.path.join(data_dir,o,x,'anat')):
                        if y.endswith('.gz'):
                            im_data.append(os.path.join(data_dir,o,x,'anat',y))

    data = []

    something = nib.load(im_data[0])

    for p, i in enumerate(im_data):
        if p % 2 != 0:
            img = nib.load(i)
            img_data = img.get_data()
            data.append(img_data)

    for i in range(data[0][1,1].shape[0]):
        plt.imshow(data[0][...,i], cmap='gray')
        plt.show()


def vis2():
    for o in os.listdir(data_dir):
        if not o.startswith('.') and os.path.isdir(os.path.join(data_dir, o)):
            for x in os.listdir(os.path.join(data_dir, o)):
                if x == 'ses-ofMcF2':
                    for y in os.listdir(os.path.join(data_dir, o, x, 'anat')):
                        if y.endswith('.gz'):
                            im_data.append(os.path.join(data_dir, o, x, 'anat', y))

    data = []

    something = nib.load(im_data[0])

    for p, i in enumerate(im_data):
        if p % 2 != 0:
            img = nib.load(i)
            img_data = img.get_data()
            data.append(img_data)

    for i in range(data[0][1, 1].shape[0]):
        plt.imshow(data[0][..., i], cmap='gray')
        plt.savefig('myimg_{}.pdf'.format(i))

vis2()
