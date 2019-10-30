import os
import imageio
import numpy as np
from matplotlib import pyplot as plt
import os

import data_loader as dl


def resample_bidsdata():
    """
    Resamples all the bidsdata and stores it to /var/tmp/resampled/

    """
    #todo if RIA change to RAS: fslswapdim input LR PA IS
    #fslhd header aufrufen

    bids_datas, file_names = dl.load_bidsdata()
    path = '/var/tmp/resampled/'

    for i in range(len(bids_datas)):
        input_image = bids_datas[i]
        file_name = file_names[i]
        resample_cmd = 'ResampleImage 3 {input} '.format(input=input_image) + path + '{output} 0.2x0.2x0.2'.format(
            output=file_name)
        os.system(resample_cmd)
        print(resample_cmd)
        # dimension_change_command = 'fslswapdim {input} LR PA IS {output}'.format(input = file_name, output = file_name)
        # os.system(dimension_change_command)
        # print(dimension_change_command)


def data_normalization(data):
    """

    :param data: shape: (x,y)
    :return: normalised input
    """

    for i in data:
        for j in range(0, i.shape[0]):
            i[j] = i[j]*1.
            i[j] = np.clip(i[j], 0, np.percentile(i[j], 99))

            i[j] = i[j] - np.amin(i[j])
            if np.amax(i[j]) != 0:
                i[j] = i[j] / np.amax(i[j])
    return data


def save_img(img_data, path):
    if not os.path.exists(path):
        os.makedirs(path)
    for j in range(img_data.shape[0]):
        print(img_data[j, ...].shape)
        plt.imshow(img_data[j, ...], cmap='gray')
        plt.savefig(os.path.join(path, 'img_{}.png'.format(j)))


def save_datavisualisation2(img_data, myocar_labels, save_folder, index_first = False, normalized = False, file_names = False):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    img_data_temp = []
    myocar_labels_temp = []

    if index_first == True:
        for i in range(0, len(img_data)):
            img_data_temp.append(np.moveaxis(img_data[i], 0, -1))
            myocar_labels_temp.append(np.moveaxis(myocar_labels[i], 0, -1))

    counter = 0
    for i, j in zip(img_data_temp[:], myocar_labels_temp[:]):
        print(counter)
        print(i.shape)
        i_patch = i[:, :, 0]
        if normalized == True:
            i_patch = i_patch*255

        j_patch = j[:, :, 0]
        j_patch = j_patch * 255
        for slice in range(1, i.shape[2]):
            temp = i[:, :, slice]
            if normalized == True:
                temp = temp * 255
            i_patch = np.hstack((i_patch, temp))


            temp = j[:, :, slice]
            temp = temp * 255
            j_patch = np.hstack((j_patch, temp))

        image = np.vstack((i_patch, j_patch))

        if file_names == False:
            imageio.imwrite(save_folder + 'mds' + '%d.png' % (counter,), image)
        else:
            imageio.imwrite(save_folder + file_names[counter] + '.png', image)
        counter = counter + 1


def save_datavisualisation3(img_data, myocar_labels, predicted_labels, save_folder, index_first = False, normalized = False, file_names = False):
    """

    :param img_data: list of arrays with shape (z, 64, 128) (if index first == true)
    :param myocar_labels: list of arrays with shape (z, 64, 128)
    :param predicted_labels: list of arrays with shape (z, 64, 128)
    :param save_folder:
    :param index_first:
    :param normalized:
    :param file_names: list of file_names

    :return:
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    img_data_temp = []
    myocar_labels_temp = []
    predicted_labels_temp = []
    if index_first == True:
        for i in range(0, len(img_data)):
            img_data_temp.append(np.moveaxis(img_data[i], 0, -1))
            myocar_labels_temp.append(np.moveaxis(myocar_labels[i], 0, -1))
            predicted_labels_temp.append(np.moveaxis(predicted_labels[i], 0, -1))
    counter = 0
    for i, j, k in zip(img_data_temp[:], myocar_labels_temp[:], predicted_labels_temp[:]):

        i_patch = i[:, :, 0]
        if normalized == True:
            i_patch = i_patch*255

        j_patch = j[:, :, 0]
        j_patch = j_patch * 255

        k_patch = k[:,:,0]
        k_patch = k_patch*255

        for slice in range(1, i.shape[2]):
            temp = i[:, :, slice]
            if normalized == True:
                temp = temp * 255
            i_patch = np.hstack((i_patch, temp))


            temp = j[:, :, slice]
            temp = temp * 255
            j_patch = np.hstack((j_patch, temp))

            temp = k[:,:,slice]
            temp = temp*255
            k_patch = np.hstack((k_patch, temp))

        image = np.vstack((i_patch, j_patch, k_patch))

        if file_names == False:
            imageio.imwrite(save_folder + 'mds' + '%d.png' % (counter,), image)
        else:
            imageio.imwrite(save_folder + file_names[counter] + '.png', image)

        counter = counter + 1



def pad_img(img):
    shape = (64, 128)
    padded = np.empty((img.shape[0], shape[0], shape[1]))
    for i in range(img.shape[0]):
        padd_y = shape[0] - img.shape[1]
        padd_x = shape[1] - img.shape[2]
        if padd_y < 0:
            something = np.empty((img.shape[0], shape[0], img.shape[2]))
            something[i] = img[i,-padd_y//2:img.shape[1]+padd_y//2,...]
            padded[i, ...] = np.pad(something[i, ...], ((0,0), (padd_x // 2, shape[1] - padd_x // 2 - img.shape[2])), 'constant')
        else:
            padded[i, ...] = np.pad(img[i, ...], ((padd_y//2, shape[0]-padd_y//2-img.shape[1]), (padd_x//2, shape[1]-padd_x//2-img.shape[2])), 'constant')
    return padded


def resize(img):
    shape = (256, 256)
    padded = np.empty((shape[0], shape[1], img.shape[2]))
    for i in range (img.shape[2]):
        padded[...,i] = tf.resize(img[..., i]/np.max(img[..., i]), output_shape = shape, mode = 'constant')       #Todo am normalizing the data here
    return padded