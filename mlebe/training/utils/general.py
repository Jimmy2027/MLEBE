import imageio
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import os
import pickle
import cv2
import mlebe.training.utils.data_loader as dl
import scipy
import scipy.ndimage
import pandas as pd


def get_image_and_mask(image, mask, shape, save_dir, slice_view, visualisation=False, blacklist_bool=True):
    if visualisation == True:
        img_unpreprocessed = []
        mask_unpreprocessed = []

    img_data = []
    mask_data = []
    img_affines = []
    mask_affines = []
    img_headers = []
    mask_headers = []
    img_file_names = []
    mask_file_names = []
    blacklisted_images = []
    blacklisted_masks = []
    if not blacklist_bool == False:
        blacklist = write_slice_blacklist()
        bl_slice_counter = 0
    for i, m in zip(image, mask):
        img_affines.append(i.affine)
        mask_affines.append(m.affine)
        img_headers.append(i.header)
        mask_headers.append(m.header)
        img_file_names.append(os.path.basename(i.file_map['image'].filename))
        mask_file_names.append(os.path.basename(m.file_map['image'].filename))
        img = i.get_data()
        img_temp = img[:, :, :]
        mask = m.get_data()
        mask_temp = mask[:, :, :]
        if slice_view == 'coronal':
            img_temp = np.moveaxis(img_temp, 1, 0)
            mask_temp = np.moveaxis(mask_temp, 1, 0)
        elif slice_view == 'axial':
            img_temp = np.moveaxis(img_temp, 2, 0)
            mask_temp = np.moveaxis(mask_temp, 2, 0)

        if visualisation == True:
            img_unpreprocessed.append(img_temp)
            mask_unpreprocessed.append(mask_temp)

            if not os.path.exists(
                    save_dir + 'visualisation/unpreprocessed/' + os.path.basename(i.file_map['image'].filename)):
                os.makedirs(save_dir + 'visualisation/unpreprocessed/' + os.path.basename(i.file_map['image'].filename))
            counter = 0
            for it in range(img_temp.shape[0]):
                plt.imshow(img_temp[it, ...], cmap='gray')
                plt.imshow(mask_temp[it, ...], alpha=0.6, cmap='Blues')
                plt.axis('off')
                plt.savefig(save_dir + 'visualisation/unpreprocessed/' + os.path.basename(
                    i.file_map['image'].filename) + '/img_{a}{it}.pdf'.format(a=counter, it=it), format='pdf')
                plt.close()
            counter += 1

        fitted_mask = arrange_mask(img_temp, mask_temp, save_dir, visualisation)

        img_temp, fitted_mask = remove_black_images(img_temp, fitted_mask, save_dir)

        img_preprocessed = preprocess(img_temp, shape, save_dir, visualisation, switched_axis=True)
        mask_preprocessed = preprocess(fitted_mask, shape, save_dir, visualisation, switched_axis=True)

        if not blacklist_bool == False:
            temp_img = {f'{idx}': img_preprocessed[idx, ...] for idx in range(img_preprocessed.shape[0])}
            temp_mask = {f'{idx}': mask_preprocessed[idx, ...] for idx in range(mask_preprocessed.shape[0])}
            blacklisted_slices = []
            for file in blacklist:
                if file.filename == os.path.basename(i.file_map['image'].filename):
                    print(file.filename, i.file_map['image'].filename)
                    blacklisted_images.append(img_preprocessed[int(file.slice), ...])
                    blacklisted_masks.append(mask_preprocessed[int(file.slice), ...])
                    blacklisted_slices.append(int(file.slice))
                    if visualisation == True:
                        if not os.path.exists(save_dir + 'visualisation/blacklisted_slices'):
                            os.makedirs(save_dir + 'visualisation/blacklisted_slices')
                        plt.imshow(temp_img['{}'.format(int(file.slice))], cmap='gray')
                        plt.imshow(temp_mask['{}'.format(int(file.slice))], alpha=0.6, cmap='Blues')
                        plt.axis('off')
                        plt.savefig(save_dir + 'visualisation/blacklisted_slices/{a}{b}.pdf'.format(a=file.filename,
                                                                                                    b=int(file.slice)),
                                    format='pdf')
                        plt.close()

                    try:
                        del temp_img['{}'.format(int(file.slice))]
                        del temp_mask['{}'.format(int(file.slice))]
                    except Exception as e:

                        print('Error for {} at: '.format(file.filename), e)
                        print('len(file): ', temp_img.shape)
                    bl_slice_counter += 1

            img_preprocessed = np.stack(
                [temp_img[f'{idx}'] for idx in range(img_preprocessed.shape[0]) if idx not in blacklisted_slices])
            mask_preprocessed = np.stack(
                [temp_mask[f'{idx}'] for idx in range(mask_preprocessed.shape[0]) if idx not in blacklisted_slices])

        img_data.append(img_preprocessed)
        mask_data.append(mask_preprocessed)
        if visualisation == True:
            if not os.path.exists(
                    save_dir + 'visualisation/preprocessed/' + os.path.basename(i.file_map['image'].filename)):
                os.makedirs(save_dir + 'visualisation/preprocessed/' + os.path.basename(i.file_map['image'].filename))
            counter = 0
            for it in range(img_preprocessed.shape[0]):
                plt.imshow(np.squeeze(img_preprocessed[it, ...]), cmap='gray')
                plt.imshow(np.squeeze(mask_preprocessed[it, ...]), alpha=0.6, cmap='Blues')
                plt.axis('off')
                plt.savefig(save_dir + 'visualisation/preprocessed/' + os.path.basename(
                    i.file_map['image'].filename) + '/img_{a}{it}.pdf'.format(a=counter, it=it), format='pdf')
                plt.close()
            counter += 1
    if not blacklist_bool == False:
        print('blacklisted {} slices'.format(bl_slice_counter))
        xfile = open(save_dir + 'blacklisted_images.pkl', 'wb')
        pickle.dump(blacklisted_images, xfile)
        xfile.close()
        yfile = open(save_dir + 'blacklisted_masks.pkl', 'wb')
        pickle.dump(blacklisted_masks, yfile)
        yfile.close()

    if visualisation:
        save_datavisualisation1(mask_data, save_dir + '/visualisation/after_rem_black_cloumns/', index_first=True,
                                normalized=True)
        save_datavisualisation1(img_unpreprocessed, save_dir + '/visualisation/', index_first=True,
                                file_names=img_file_names, file_name_header='unpro_')
        save_datavisualisation1(img_data, save_dir + '/visualisation/', index_first=True, normalized=True,
                                file_names=img_file_names, file_name_header='prepr_')
        save_datavisualisation1(mask_unpreprocessed, save_dir + '/visualisation/', index_first=True,
                                file_names=mask_file_names, file_name_header='unpro_', normalized=True)
        save_datavisualisation1(mask_data, save_dir + '/visualisation/', index_first=True, normalized=True,
                                file_names=mask_file_names, file_name_header='prepr_')

    # save_images(img_data, mask_data, img_file_names, save_dir) #with this line can save all the images with the mask to create a blacklist

    return img_data, mask_data, img_affines, img_headers, img_file_names, mask_affines, mask_headers


def remove_outliers(image, visualisation=False, save_dir=''):
    from scipy import ndimage
    markers = ndimage.label(image)[0]
    if len(np.unique(markers)) > 2:
        l, counts = np.unique(markers, return_counts=True)
        brain_label = l[np.argsort(-counts)[1]]
        new = np.where(markers == brain_label, 1, 0)
        if visualisation == True:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for slice in range(image.shape[0]):
                if len(np.unique(markers[slice])) > 2:
                    plt.figure()
                    plt.axis('off')
                    plt.subplot(1, 3, 1)
                    plt.imshow(image[slice])
                    plt.subplot(1, 3, 2)
                    plt.imshow(markers[slice])
                    plt.subplot(1, 3, 3)
                    plt.imshow(new[slice])
                    plt.savefig(save_dir + '{}.pdf'.format(slice), format='pdf')
                    plt.close()
        return new
    else:
        return image


def save_images(images, mask, img_file_names, save_dir):
    """
    can save all the images with the mask to create a blacklist
    :param images:
    :param mask:
    :param img_file_names:
    :param save_dir:
    :return:
    """
    counter = 0
    for im, ma in zip(images, mask):
        if not os.path.exists(save_dir + '/images/{name}'.format(name=img_file_names[counter])):
            os.makedirs(save_dir + '/images/{name}'.format(name=img_file_names[counter]))
        for i in range(im.shape[0]):
            plt.imshow(im[i, ...], cmap='gray')
            plt.imshow(ma[i, ...], alpha=0.6, cmap='Blues')
            plt.savefig(save_dir + '/images/{name}/{i}.pdf'.format(name=img_file_names[counter], i=i), format='pdf')
            plt.close()
        counter += 1


def arrange_mask(img, mask, save_dir=False, visualisation=False):
    """
    Sets the mask to zero where the image is zero
    """

    new_mask = mask[:, :, :]

    new_mask[img == 0] = 0

    fixed_mask = new_mask[:, :, :]

    structure = [[1, 0, 1], [1, 1, 1], [0, 1, 0]]

    for i in range(new_mask.shape[0]):
        fixed_mask[i] = scipy.ndimage.morphology.binary_fill_holes(new_mask[i], structure=structure)

    if visualisation == True:
        save_datavisualisation([img, mask, new_mask, fixed_mask], save_dir + 'visualisation/arrange_mask/')

    return fixed_mask


def remove_black_images(img, mask, save_dir=None, visualisation=False):
    new_img = img[:, :, :]
    new_mask = mask[:, :, :]
    if visualisation:
        before_img = img[:, :, :]
        before_mask = mask[:, :, :]

    counter = 0
    if not img.shape[0] == 0:
        for z in range(img.shape[0]):

            if len(np.unique(img[z, ...])) == 1:
                # temp_path = check_path(save_dir + '/visualisation/remove_black_img/', 'removed_{}'.format(z))
                # plt.imshow(img[z, ...])
                # plt.title(str(np.sum(np.concatenate(img[z, ...]))))
                # plt.savefig(temp_path)
                new_img = np.delete(new_img, z - counter, 0)
                new_mask = np.delete(new_mask, z - counter, 0)
                counter += 1

        if new_img.shape[0] == 0:
            save_datavisualisation1(before_img, save_dir + '/visualisation/gfhsghsdh/',
                                    index_first=True, file_name_header='img')
            return None, None

        if visualisation == True:
            save_datavisualisation2(before_img, new_img, save_dir + '/visualisation/remove_black_img/',
                                    index_first=True, file_name_header='img')
            save_datavisualisation2(before_mask, new_mask, save_dir + '/visualisation/remove_black_img/',
                                    index_first=True, file_name_header='mask')

    return new_img, new_mask


def remove_black_masks(img, mask, save_dir=None, visualisation=False):
    if visualisation == True:
        before_img = list(img)
        before_mask = list(mask)
    idxs1 = []
    idxs2 = []
    for n in range(len(img)):
        for i in range(mask[n].shape[0]):
            if np.max(mask[n][i]) == 0:
                idx1 = i

            else:
                break
        img[n] = img[n][idx1 + 1:, ...]
        mask[n] = mask[n][idx1 + 1:, ...]
        idxs1.append(idx1 + 1)

    for n in range(len(img)):
        for i in range(mask[n].shape[0] - 1, -1, -1):
            if np.max(mask[n][i]) == 0:
                idx2 = i

            else:
                break
        img[n] = img[n][:idx2, ...]
        mask[n] = mask[n][:idx2, ...]
        idxs2.append(idx2)

    if visualisation == True:
        save_datavisualisation2(before_img, img, save_dir + '/visualisation/remove_black_mask/', index_first=True,
                                normalized=True, idx1=idxs1, idx2=idxs2)
        save_datavisualisation2(before_mask, mask, save_dir + '/visualisation/remove_black_mask/', index_first=True,
                                normalized=True, idx1=idxs1, idx2=idxs2)

    return img, mask


def remove_black_columns(img, save_dir=None, visualisation=False):
    """
    Looks at 20th slice and removes all the columns at the border of the images that are 0
    :param img:
    :return:
    """

    for i in range(img[0, ...].shape[1]):
        if max(img[0, :, i]) > 0:
            id1 = i
            break

    for i in range(img[0, ...].shape[1] - 1, -1, -1):
        if max(img[0, :, i]) > 0:
            id2 = i
            break

    new_img = img[:, :, id1:id2]

    if visualisation:
        save_datavisualisation2(new_img, img, save_dir + '/visualisation/remove_black_columns/', normalized=True,
                                index_first=True)

    return new_img, id1, id2


def get_data(data, shape, save_dir, visualisation=False, verbose=False):
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
        save_datavisualisation1(unpreprocessed, save_dir + '/visualisaion/', index_first=True, file_names=file_names,
                                file_name_header='unpro_')
        save_datavisualisation1(img_data, save_dir + '/visualisaion/', index_first=True, normalized=True,
                                file_names=file_names, file_name_header='prepr_')

    return img_data, affines, headers, file_names


def preprocess(img, shape, slice_view, save_dir=None, visualisation=False, switched_axis=False, normalize=True):
    """
    This preprocessing function reshapes the volume into the desired shape by first zero-padding the smaller dimension to the same size as the bigger one and then reshaping the image with cv2.resize.

    - moves axis such that (x,y,z) becomes (z,x,y)
    - transforms the image such that shape is (z,shape). If one dimension is bigger than shape -> downscale, if one dimension is smaller -> zero-pad
    - normalizes the data
    :param img: img with shape (x,y,z)
    :return: img with shape (z,shape)
    """
    if switched_axis == False:
        if slice_view == 'coronal':
            img = np.moveaxis(img, 1, 0)
        elif slice_view == 'axial':
            img = np.moveaxis(img, 2, 0)

    img_data = pad_and_reshape_img(img, shape)
    if normalize:
        img_data = data_normalization(img_data)

    return img_data


def resample_bidsdata(path):
    """
    Resamples all the bidsdata and stores it to path
    AND changes dimensions to RAS
    """
    # fslhd header aufrufen

    if not os.path.exists(path):
        os.makedirs(path)

    bids_datas, file_names = dl.load_bidsdata()

    for i in range(len(bids_datas)):
        input_image = bids_datas[i]
        file_name = file_names[i]
        resample_cmd = 'ResampleImage 3 {input} '.format(input=input_image) + path + '{output} 0.2x0.2x0.2'.format(
            output=file_name)
        os.system(resample_cmd)
        print(resample_cmd)
        dimension_change_command = 'fslswapdim ' + path + '{input} LR PA IS '.format(
            input=file_name) + path + '{output}'.format(output=file_name)
        os.system(dimension_change_command)
        print(dimension_change_command)


def data_normalization(data):
    """

    :param data: shape: (y, x)
    :return: normalised input
    """
    data = data * 1.
    data = np.clip(data, 0, np.percentile(data, 99))

    data = data - np.amin(data)
    if np.amax(data) != 0:
        data = data / np.amax(data)
    return data


def save_img(img_data, path):
    if not os.path.exists(path):
        os.makedirs(path)
    for j in range(img_data.shape[0]):
        print(img_data[j, ...].shape)
        plt.imshow(img_data[j, ...], cmap='gray')
        plt.savefig(os.path.join(path, 'img_{}.pdf'.format(j)), format='pdf')


def save_datavisualisation1(img_data, save_folder, index_first=True, normalized=False, file_names=False,
                            file_name_header=False):
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

        i_patch = i[:, :, 0]
        if normalized == True:
            i_patch = i_patch * 255

        for slice in range(1, i.shape[2]):
            temp = i[:, :, slice]
            if normalized == True:
                temp = temp * 255
            i_patch = np.hstack((i_patch, temp))

        image = i_patch

        if file_names == False:
            i = 0
            while os.path.exists(save_folder + 'mds_{}_'.format(i) + '%d.png' % (counter,)):
                i += 1
            imageio.imwrite(save_folder + 'mds_{}_'.format(i) + '%d.png' % (counter,), image)
        else:
            if file_name_header == False:
                i = 0
                while os.path.exists(save_folder + file_names[counter] + '{}.png'.format(i)):
                    i += 1
                imageio.imwrite(save_folder + file_names[counter] + '{}.png'.format(i), image)
            else:
                i = 0
                while os.path.exists(save_folder + file_name_header + file_names[counter] + '{}.png'.format(i)):
                    i += 1
                imageio.imwrite(save_folder + file_name_header + file_names[counter] + '{}.png'.format(i), image)
        counter = counter + 1


def save_datavisualisation2(img_data, myocar_labels, save_folder, file_name_header=False, index_first=True,
                            normalized=False, file_names=False, idx1=None, idx2=None):
    if normalized == False:
        img_data = data_normalization(img_data)
        myocar_labels = data_normalization(myocar_labels)
        normalized = True
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if not type(img_data) == list:
        temp = list(img_data)
        img_data = []
        img_data.append(temp)

    if not type(myocar_labels) == list:
        temp = list(myocar_labels)
        myocar_labels = []
        myocar_labels.append(temp)

    img_data_temp = []
    myocar_labels_temp = []

    if index_first == True:
        for i in range(0, len(img_data)):
            img_data_temp.append(np.moveaxis(img_data[i], 0, -1))
            myocar_labels_temp.append(np.moveaxis(myocar_labels[i], 0, -1))

    counter = 0
    for i, j in zip(img_data_temp[:], myocar_labels_temp[:]):
        if i.shape != j.shape:  # j need to be bigger than i     #todo this needs to be generalised
            i = np.pad(i, (((j.shape[0] - i.shape[0]) // 2, j.shape[0] - i.shape[0] - (j.shape[0] - i.shape[0]) // 2),
                           ((j.shape[1] - i.shape[1]) // 2, j.shape[1] - i.shape[1] - (j.shape[1] - i.shape[1]) // 2),
                           (0, 0)), mode='constant', constant_values=0.5)
            if idx1 == None:
                j = np.pad(j, ((0, 0), (0, 0), (
                    (i.shape[2] - j.shape[2]) // 2, i.shape[2] - j.shape[2] - (i.shape[2] - j.shape[2]) // 2)),
                           mode='constant', constant_values=0.5)
            else:
                j = np.pad(j, ((0, 0), (0, 0), (idx1[counter], idx2[counter])), mode='constant', constant_values=0.5)

        i_patch = i[:, :, 0]
        if normalized == True:
            i_patch = i_patch * 255

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
            i = 0
            while os.path.exists(save_folder + 'mds_{}_'.format(i) + '%d.png' % (counter,)):
                i += 1
            imageio.imwrite(save_folder + 'mds_{}_'.format(i) + '%d.png' % (counter,), image)
        else:
            if file_name_header == False:
                i = 0
                while os.path.exists(save_folder + file_names[counter] + '{}.png'.format(i)):
                    i += 1
                imageio.imwrite(save_folder + file_names[counter] + '{}.png'.format(i), image)
            else:
                i = 0
                while os.path.exists(save_folder + file_name_header + file_names[counter] + '{}.png'.format(i)):
                    i += 1
                imageio.imwrite(save_folder + file_name_header + file_names[counter] + '{}.png'.format(i), image)
        counter = counter + 1


def save_datavisualisation3(img_data, myocar_labels, predicted_labels, save_folder, index_first=False, normalized=False,
                            file_names=False):
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
    if normalized == False:
        img_data = data_normalization(img_data)
        myocar_labels = data_normalization(myocar_labels)
        normalized = True

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if not type(img_data) == list:
        temp = list(img_data)
        img_data = []
        img_data.append(temp)

    if not type(myocar_labels) == list:
        temp = list(myocar_labels)
        myocar_labels = []
        myocar_labels.append(temp)

    if not type(predicted_labels) == list:
        temp = list(predicted_labels)
        predicted_labels = []
        predicted_labels.append(temp)

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
            i_patch = i_patch * 255

        j_patch = j[:, :, 0]
        j_patch = j_patch * 255

        k_patch = k[:, :, 0]
        k_patch = k_patch * 255

        for slice in range(1, i.shape[2]):
            temp = i[:, :, slice]
            if normalized == True:
                temp = temp * 255
            i_patch = np.hstack((i_patch, temp))

            temp = j[:, :, slice]
            temp = temp * 255
            j_patch = np.hstack((j_patch, temp))

            temp = k[:, :, slice]
            temp = temp * 255
            k_patch = np.hstack((k_patch, temp))

        image = np.vstack((i_patch, j_patch, k_patch))

        if file_names == False:
            path = check_path(save_folder, 'img' + '%d' % (counter,))
            imageio.imwrite(path + '.png', image)
        else:
            path = check_path(save_folder, file_names[counter])
            imageio.imwrite(path + '.png', image)

        counter = counter + 1


def save_datavisualisation(images, save_folder, file_name_header=False, normalized=False, file_names=False):
    """

    :param images: a list of lists of sliced images, where the slice index is in the first dimension
    :param save_folder:
    :param file_name_header:
    :param normalized:
    :param file_names:
    :return:
    """
    if normalized == False:
        for l in range(len(images)):
            for i in range(len(images[l])):
                images[l][i] = data_normalization(images[l][i])

            if not type(images[l]) is type(images):
                temp = []
                temp.append(images[l])
                images[l] = temp

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    counter = 0
    for img in range(len(images[0])):  # number of images that will be saved at the end

        patches = []
        for list in range(len(images)):
            patch = images[list][img][0, :, :] * 255
            for slice in range(1, images[list][img].shape[0]):
                temp = images[list][img][slice, :, :] * 255
                patch = np.hstack((patch, temp))

            patches.append(patch)

        patch = patches[0]
        for i in range(1, len(patches)):
            patch = np.vstack((patch, patches[i]))

        image = np.vstack(patches)

        if file_names == False:
            i = 0
            while os.path.exists(save_folder + 'mds_{}_'.format(i) + '%d.png' % (counter,)):
                i += 1
            plt.figure(figsize=(len(images[0][0]), len(images[0])))
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            plt.savefig(save_folder + 'mds_{}_'.format(i) + '{}.pdf'.format(counter), format="pdf", dpi=300,
                        bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            if file_name_header == False:
                i = 0
                while os.path.exists(save_folder + file_names[counter] + '{}.png'.format(i)):
                    i += 1
                plt.figure(figsize=(len(images[0][0]), len(images[0])))
                plt.imshow(image, cmap='gray')
                plt.axis('off')
                plt.savefig(save_folder + file_names[counter] + '{}.pdf'.format(i), format="pdf", dpi=300,
                            bbox_inches='tight', pad_inches=0)
                plt.close()
            else:
                i = 0
                while os.path.exists(save_folder + file_name_header + file_names[counter] + '{}.png'.format(i)):
                    i += 1
                plt.figure(figsize=(len(images[0][0]), len(images[0])))
                plt.imshow(image, cmap='gray')
                plt.axis('off')
                plt.savefig(save_folder + file_name_header + file_names[counter] + '{}.pdf'.format(i), format="pdf",
                            dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close()
        counter += 1


def save_datavisualisation_plt(images, save_folder, file_name_header=False, normalized=False, file_names=False,
                               row_titles=False, figure_title=False, slice_titles=False):
    """

    :param images: a list of lists of sliced images, where the slice index is in the first dimension
    :param row_titles:
    :param save_folder:
    :param file_name_header:
    :param normalized:
    :param file_names:
    :return:
    """
    if normalized == False:
        for l in range(len(images)):
            for i in range(len(images[l])):
                images[l][i] = data_normalization(images[l][i])

            if not type(images[l]) is type(images):
                temp = []
                temp.append(images[l])
                images[l] = temp

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    counter = 0
    for img in range(len(images[0])):  # number of images that will be saved at the end

        nrow = len(images)
        ncol = images[0][img].shape[0]

        plt.switch_backend('agg')
        figure = plt.figure(figsize=(ncol + 1, nrow + 1))

        anno_opts = dict(xy=(0.5, 0.5), xycoords='axes fraction',
                         va='center', ha='center')
        if not figure_title == False:
            figure.suptitle(figure_title, fontsize=20)

        gs = gridspec.GridSpec(nrow + 1, ncol + 1, wspace=0.6, hspace=0.2, top=1. - 0.5 / (nrow + 1),
                               bottom=0.5 / (nrow + 1), left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))

        for list in range(len(images)):
            for slice in range(images[list][img].shape[0] + 1):

                i_col = slice
                i_row = list
                ax = plt.subplot(gs[i_row, i_col])
                if i_col == 0:
                    if not row_titles == False:
                        if not row_titles[i_row] is None:
                            ax.annotate(row_titles[i_row], **anno_opts)
                    plt.axis('off')
                else:
                    if not slice_titles == False:
                        if not slice_titles[list] is None:
                            ax.set_title(slice_titles[list][img][slice - 1], fontdict={'fontsize': 8})
                    image = images[list][img][slice - 1, :, :] * 255
                    plt.imshow(image.repeat(2, axis=0).repeat(2, axis=1), cmap='gray')
                    plt.axis('off')

        if file_names == False:
            i = 0
            while os.path.exists(save_folder + 'mds_{}_'.format(i) + '%d.pdf' % (counter,)):
                i += 1
            plt.ioff()
            plt.switch_backend('agg')
            figure.savefig(save_folder + 'mds_{}_'.format(i) + '%d.pdf' % (counter,), format='pdf')
            plt.close(figure)
        else:
            if file_name_header == False:
                i = 0
                while os.path.exists(save_folder + file_names[counter] + '{}.pdf'.format(i)):
                    i += 1
                plt.ioff()
                plt.switch_backend('agg')
                figure.savefig(save_folder + file_names[counter] + '{}.pdf'.format(i), format='pdf')
                plt.close(figure)
            else:
                i = 0
                while os.path.exists(save_folder + file_name_header + file_names[counter] + '{}.pdf'.format(i)):
                    i += 1
                plt.ioff()
                plt.switch_backend('agg')
                plt.tight_layout()
                figure.savefig(save_folder + file_name_header + file_names[counter] + '{}.pdf'.format(i), format='pdf')

                plt.close(figure)
        counter = counter + 1


def save_datavisualisation_plt_subsubplot(images, save_folder, file_name_header=False, normalized=False,
                                          file_names=False, row_titles=False, slice_titles=False):
    """

    :param images: a list of lists of sliced images, where the slice index is in the first dimension
    :param row_titles:
    :param save_folder:
    :param file_name_header:
    :param normalized:
    :param file_names:
    :return:
    """
    if normalized == False:
        for l in range(len(images)):
            for i in range(len(images[l])):
                images[l][i] = data_normalization(images[l][i])

            if not type(images[l]) is type(images):
                temp = []
                temp.append(images[l])
                images[l] = temp

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for img in range(len(images[0])):  # number of images that will be saved at the end
        counter = 0

        nrow_outer = len(images)
        ncol_outer = 1
        ncol_inner = images[0][img].shape[0]
        nrow_inner = 1
        plt.switch_backend('agg')
        figure = plt.figure(figsize=(ncol_inner + 1, nrow_outer + 1))

        outer_gs = gridspec.GridSpec(nrow_outer, ncol_outer, wspace=1, hspace=0.25, top=1. - 0.5 / (nrow_outer + 1),
                                     bottom=0.5 / (nrow_outer + 1),
                                     left=0.5 / (ncol_outer + 1), right=1 - 0.5 / (ncol_outer + 1))

        for list in range(len(images)):
            ax_outer = plt.subplot(outer_gs[list])
            ax_outer.set_title('hihihi', fontdict={'fontsize': 100})
            inner_gs = gridspec.GridSpecFromSubplotSpec(nrow_inner, ncol_inner, subplot_spec=outer_gs[list], wspace=1,
                                                        hspace=0.25)
            for slice in range(images[list][img].shape[0]):
                i_col = slice
                i_row = list
                ax = plt.subplot(inner_gs[i_col])
                if not slice_titles == False:
                    if not slice_titles[list] is None:
                        plt.title(slice_titles[list][img][slice])
                plt.imshow(images[list][img][slice, :, :] * 255, cmap='gray')
                plt.axis('off')

        if file_names == False:
            i = 0
            while os.path.exists(save_folder + 'mds_{}_'.format(i) + '%d.pdf' % (counter,)):
                i += 1
            plt.ioff()
            plt.switch_backend('agg')
            figure.savefig(save_folder + 'mds_{}_'.format(i) + '%d.pdf' % (counter,), format='pdf')
            plt.close(figure)
        else:
            if file_name_header == False:
                i = 0
                while os.path.exists(save_folder + file_names[counter] + '{}.pdf'.format(i)):
                    i += 1
                plt.ioff()
                plt.switch_backend('agg')
                figure.savefig(save_folder + file_names[counter] + '{}.pdf'.format(i), format='pdf')
                plt.close(figure)
            else:
                i = 0
                while os.path.exists(save_folder + file_name_header + file_names[counter] + '{}.pdf'.format(i)):
                    i += 1
                plt.ioff()
                plt.switch_backend('agg')
                figure.savefig(save_folder + file_name_header + file_names[counter] + '{}.pdf'.format(i), format='pdf')

                plt.close(figure)
            counter = counter + 1

def pad_and_reshape_img(img, shape):
    """
    The preprocessing function reshapes the volume into the desired shape by first zero-padding the smaller dimension to the same size as the bigger one and then reshaping the image with cv2.resize.
    """

    padded = np.empty((img.shape[0], shape[0], shape[1]))

    if img.shape[1] > img.shape[2]:
        for i in range(img.shape[0]):
            padd = img.shape[1] - img.shape[2]
            temp_padded = np.pad(img[i, ...], ((0, 0), (padd // 2, img.shape[1] - padd // 2 - img.shape[2])),
                                 'constant')
            padded[i] = cv2.resize(temp_padded, (shape[1], shape[0]))

    elif img.shape[1] < img.shape[2]:
        for i in range(img.shape[0]):
            padd = img.shape[2] - img.shape[1]
            temp_padded = np.pad(img[i, ...], ((padd // 2, img.shape[2] - padd // 2 - img.shape[1]), (0, 0)),
                                 'constant')
            padded[i] = cv2.resize(temp_padded, (shape[1], shape[0]))
    else:
        for i in range(img.shape[0]):
            temp = cv2.resize(img[i], (shape[1], shape[0]))
            padded[i] = temp

    return padded

def pad_img(img,shape):
    shape_diffs = [int(np.ceil((i_s - d_s))) for d_s, i_s in zip(img.shape, shape)]
    shape_diffs = np.maximum(shape_diffs, 0)
    pad_sizes = [(int(np.ceil(s / 2.)), int(np.floor(s / 2.))) for s in shape_diffs]
    padded = np.pad(img, pad_sizes, mode='constant')
    return padded

def write_slice_blacklist():
    blacklist = []

    class blacklist_elem:
        def __init__(self, filename, slice):
            self.filename = filename
            self.slice = slice

    slice_blacklist_path = os.path.expanduser('~/src/MLEBE/mlebe/Blacklist/mlebe_slice_blacklist.txt')
    if not os.path.isfile(slice_blacklist_path):
        print('\n\n Need to create blacklist file! \n\n')
        os.system(
            'tag -f mlebe_blacklist > ' + os.path.expanduser('~/src/MLEBE/mlebe/Blacklist/mlebe_slice_blacklist.txt'))
    with open(slice_blacklist_path) as blacklist_file:
        for line in blacklist_file:
            line = line.replace('.png', '')
            line = line.split('/')
            line = line[-2:]
            blacklist.append(blacklist_elem(line[0], line[1]))
    return blacklist


def write_blacklist(blacklist_dir):
    blacklist = []

    class blacklist_elem:
        def __init__(self, subject, session):
            self.subj = subject
            self.sess = session

    for file in os.listdir(blacklist_dir):
        if file.endswith('.pdf'):
            temp = file.replace('.pdf', '')
            temp2 = temp.split('_')
            blacklist.append(blacklist_elem(temp2[0], temp2[1]))
    return blacklist


def check_path(path, filename='img', format='.png'):
    if not os.path.exists(path):
        os.makedirs(path)
    i = 0
    while os.path.isfile(path + filename + '{}'.format(i) + format):
        i += 1

    return path + filename + '{}'.format(i)


def corr(img1, img2):
    d = {'img1': img1.reshape(-1), 'img2': img2.reshape(-1)}
    data = pd.DataFrame(data=d)
    corr = data.corr(method='pearson')

    return corr.values[0][1]


def compute_correlation(images1, images2, images3, save_dir):
    d = {'x_test': images1.reshape(-1), 'y_test': images2.reshape(-1), 'y_pred': images3.reshape(-1)}
    data = pd.DataFrame(data=d)
    corr = data.corr(method='pearson')

    fig, ax = plt.subplots()

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    ax.table(cellText=corr.values, rowLabels=corr.columns, colLabels=corr.columns, loc='center')

    # fig.tight_layout()

    plt.savefig(save_dir + 'correlation.pdf', format='pdf')
    plt.close()
