import os
import imageio
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas
import cv2
import data_loader as dl


def get_data(data, shape, save_dir,  visualisation = False, verbose = False):
    """

    :param data: list of Nifit1Images
    :param shape: Tuple (y,x): shape of the images after preprocessing
    :param save_dir: directory where images will be saved if visualisation is True
    :param visualisation: Bool: if True, unpreprocessed images are saved
    :return: img_data, affines, headers, file_names
    """
    if visualisation == True:
        unpreprocessed = []
    img_data = []
    affines = []
    headers = []
    file_names = []
    for i in data:
        affines.append(i.affine)
        headers.append(i.header)
        file_names.append(os.path.basename(i.file_map['image'].filename))
        temp = i.get_data()
        if visualisation == True:
            unpreprocessed.append(np.moveaxis(temp, 2, 0))
        preprocessed = preprocess(temp, shape)
        img_data.append(preprocessed)

    if visualisation == True:
        save_datavisualisation1(unpreprocessed, save_dir + '/visualisaion/', index_first= True, file_names=file_names, file_name_header= 'unpro_')
        save_datavisualisation1(img_data, save_dir + '/visualisaion/', index_first= True,normalized= True  ,file_names=file_names, file_name_header= 'prepr_')

    return img_data, affines, headers, file_names

def preprocess(img, shape):
    """
    - moves axis such that (x,y,z) becomes (z,x,y)
    - transforms the image such that shape is (z,shape). If one dimension is bigger than shape -> downscale, if one dimension is smaller -> zero-pad
    - normalizes the data
    :param img: img with shape (x,y,z)
    :return: img with shape (z,shape)
    """
    temp = np.moveaxis(img, 2, 0)
    img_data = pad_img(temp, shape)
    img_data = data_normalization(img_data)

    return img_data

def resample_bidsdata(path):
    """
    Resamples all the bidsdata and stores it to /var/tmp/resampled/
    AND changes dimensions to RAS
    """
    #fslhd header aufrufen

    bids_datas, file_names = dl.load_bidsdata()

    for i in range(len(bids_datas)):
        input_image = bids_datas[i]
        file_name = file_names[i]
        resample_cmd = 'ResampleImage 3 {input} '.format(input=input_image) + path + '{output} 0.2x0.2x0.2'.format(
            output=file_name)
        os.system(resample_cmd)
        print(resample_cmd)
        dimension_change_command = 'fslswapdim ' + path + '{input} LR PA IS '.format(input = file_name) + path+'{output}'.format(output = file_name)
        os.system(dimension_change_command)
        print(dimension_change_command)


def data_normalization(data):
    """

    :param data: shape: (y, x)
    :return: normalised input
    """

    for i in range(data.shape[0]):
        data[i] = data[i]*1.
        data[i] = np.clip(data[i], 0, np.percentile(data[i], 99))

        data[i] = data[i] - np.amin(data[i])
        if np.amax(data[i]) != 0:
            data[i] = data[i] / np.amax(data[i])
    return data


def save_img(img_data, path):
    if not os.path.exists(path):
        os.makedirs(path)
    for j in range(img_data.shape[0]):
        print(img_data[j, ...].shape)
        plt.imshow(img_data[j, ...], cmap='gray')
        plt.savefig(os.path.join(path, 'img_{}.png'.format(j)))

def save_datavisualisation1(img_data, save_folder, index_first = False, normalized = False, file_names = False, file_name_header = False):
    """

    :param img_data:
    :param save_folder:
    :param index_first:
    :param normalized:
    :param file_names:
    :param file_name_header: i.e. 'preprocessed_' at the beginning of each file_name
    :return:
    """

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    img_data_temp = []

    if index_first == True:
        for i in range(0, len(img_data)):
            img_data_temp.append(np.moveaxis(img_data[i], 0, -1))

    counter = 0
    for i in img_data_temp[:]:
        print(counter)
        print(i.shape)
        i_patch = i[:, :, 0]
        if normalized == True:
            i_patch = i_patch*255

        for slice in range(1, i.shape[2]):
            temp = i[:, :, slice]
            if normalized == True:
                temp = temp * 255
            i_patch = np.hstack((i_patch, temp))

        image = i_patch

        if file_names == False:
            imageio.imwrite(save_folder + 'mds' + '%d.png' % (counter,), image)
        else:
            if file_name_header == False:
                imageio.imwrite(save_folder + file_names[counter] + '.png', image)
            else:
                imageio.imwrite(save_folder + file_name_header +file_names[counter] + '.png', image)
        counter = counter + 1


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

    :param img_data: list of arrays with shape (z, y, x) (if index first == true)
    :param myocar_labels: list of arrays with shape (z, y, x)
    :param predicted_labels: list of arrays with shape (z, y, x)
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



def pad_img(img, shape):
    """
    Reshapes input image to shape. If input shape is bigger -> resize, if it is smaller -> zero-padd
    :param img:
    :param shape: shape in (y,x)
    :return:
    """

    padded = np.empty((img.shape[0], shape[0], shape[1]))
    padd_y = shape[0] - img.shape[1]
    padd_x = shape[1] - img.shape[2]
    for i in range(img.shape[0]):
        if padd_x < 0 and padd_y < 0:
            temp = cv2.resize(img[i], (shape[1], shape[0]))
            padded[i] = temp
        elif padd_y < 0:
            temp = cv2.resize(img[i], (img[i].shape[1], shape[0])) #cv2.resize takes shape in form (x,y)!!!
            something = np.empty((img.shape[0], shape[0], img.shape[2]))
            something[i] = temp
            padded[i, ...] = np.pad(something[i, ...], ((0,0), (padd_x // 2, shape[1] - padd_x // 2 - img.shape[2])), 'constant')
        elif padd_x < 0:
            temp = cv2.resize(img[i], (shape[1], img[i].shape[0]))
            padded[i] = np.pad(temp, ((padd_y//2, shape[0]-padd_y//2-img.shape[1]), (0,0)), 'constant')
        else:
            padded[i, ...] = np.pad(img[i, ...], ((padd_y//2, shape[0]-padd_y//2-img.shape[1]), (padd_x//2, shape[1]-padd_x//2-img.shape[2])), 'constant')
    return padded


def write_blacklist(blacklist_dir):
    blacklist = []
    for file in os.listdir(blacklist_dir):
        temp = file.replace('.pdf', '')
        temp2 = temp.split('_')
        blacklist.append(temp2)
    return blacklist

