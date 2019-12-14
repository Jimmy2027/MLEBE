import os
import imageio
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas
import cv2
import data_loader as dl
import scipy


def get_image_and_mask(image, mask, shape, save_dir, remove_black_labels_and_columns, slice_view, visualisation = False):
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
    for i, m in zip(image, mask):
        img_affines.append(i.affine)
        mask_affines.append(m.affine)
        img_headers.append(i.header)
        mask_headers.append(m.header)
        img_file_names.append(os.path.basename(i.file_map['image'].filename))
        mask_file_names.append(os.path.basename(m.file_map['image'].filename))
        img = i.get_data()
        img_temp = img[:,:,:]
        mask = m.get_data()
        mask_temp = mask[:, :, :]
        if slice_view == 'coronal':
            img_temp = np.moveaxis(img_temp, 1, 0)
            mask_temp = np.moveaxis(mask_temp, 1, 0)
        elif slice_view == 'transverse':
            img_temp = np.moveaxis(img_temp, 2, 0)
            mask_temp = np.moveaxis(mask_temp, 2, 0)

        if visualisation == True:
            img_unpreprocessed.append(img_temp)
            mask_unpreprocessed.append(mask_temp)



        fitted_mask = arrange_mask(img_temp, mask_temp, save_dir, visualisation)


        img_temp, fitted_mask = remove_black_images(img_temp, fitted_mask, save_dir, visualisation= visualisation)
            # if img_temp is None:
            #     continue
            # img_temp, id1, id2 = remove_black_columns(img_temp, save_dir, visualisation)
            # mask_temp = mask_temp[:,:,id1:id2]

        img_preprocessed = preprocess(img_temp, shape, save_dir, visualisation, switched_axis= True)
        mask_preprocessed = preprocess(fitted_mask, shape, save_dir, visualisation, switched_axis = True)
        img_data.append(img_preprocessed)
        mask_data.append(mask_preprocessed)

    if visualisation:
        save_datavisualisation1(mask_data, save_dir + '/visualisation/after_rem_black_cloumns/', index_first= True, normalized= True)

    # if remove_black_labels_and_columns:
    #     img_data, mask_data = remove_black_masks(img_data, mask_data, save_dir= save_dir, visualisation=visualisation)


    if visualisation == True:
        if not os.path.exists(save_dir + 'visualisation/preprocessed/'):
            os.makedirs(save_dir + 'visualisation/preprocessed/')
        counter = 0
        for im, ma in zip(img_data, mask_data):
            for i in range(im.shape[0]):
                plt.imshow(np.squeeze(im[i, ...]), cmap='gray')
                plt.imshow(np.squeeze(ma[i, ...]), alpha=0.3, cmap='Blues')
                plt.savefig(save_dir + 'visualisation/preprocessed/img_{a}{i}'.format(a=counter,i=i))
                plt.close()
            counter += 1

        if not os.path.exists(save_dir + 'visualisation/unpreprocessed/'):
            os.makedirs(save_dir + 'visualisation/unpreprocessed/')
        counter = 0
        for im, ma in zip(img_unpreprocessed, mask_unpreprocessed):
            for i in range(im.shape[0]):
                plt.imshow(im[i, ...], cmap='gray')
                plt.imshow(ma[i, ...], alpha=0.3, cmap='Blues')
                plt.savefig(save_dir + 'visualisation/unpreprocessed/img_{a}{i}'.format(a=counter,i=i))
                plt.close()
            counter += 1

        save_datavisualisation1(img_unpreprocessed, save_dir + '/visualisation/', index_first= True, file_names=img_file_names, file_name_header= 'unpro_')
        save_datavisualisation1(img_data, save_dir + '/visualisation/', index_first= True, normalized= True  ,file_names=img_file_names, file_name_header= 'prepr_')
        save_datavisualisation1(mask_unpreprocessed, save_dir + '/visualisation/', index_first= True, file_names=mask_file_names, file_name_header= 'unpro_', normalized= True)
        save_datavisualisation1(mask_data, save_dir + '/visualisation/', index_first= True, normalized= True,file_names=mask_file_names, file_name_header= 'prepr_')

    return img_data, mask_data, img_affines, img_headers, img_file_names, mask_affines, mask_headers


def arrange_mask(img, mask, save_dir, visualisation = False):

    new_mask = mask[:,:,:]

    new_mask[img == 0] = 0

    fixed_mask = new_mask[:, :, :]

    structure = [[1,0,1], [1,1,1], [0,1,0]]

    for i in range(new_mask.shape[0]):
        fixed_mask[i] = scipy.ndimage.morphology.binary_fill_holes(new_mask[i], structure=structure)

    if visualisation == True:
        save_datavisualisation3(img,new_mask, fixed_mask, save_dir + 'visualisation/arrange_mask/', index_first= True)

    return fixed_mask





def remove_black_images(img, mask, save_dir = None, visualisation = False):

    new_img = img[:, :, :]
    new_mask = mask[:, :, :]
    if visualisation:
        before_img = img[:,:,:]
        before_mask = mask[:,:,:]

    counter = 0
    if not img.shape[0] == 0:
        for z in range(img.shape[0]):

            if np.max(img[z,...]) == 0:
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
            save_datavisualisation2(before_img, new_img, save_dir + '/visualisation/remove_black_img/', index_first= True, file_name_header= 'img')
            save_datavisualisation2(before_mask, new_mask, save_dir + '/visualisation/remove_black_img/', index_first=True, file_name_header= 'mask')


    return new_img, new_mask


def remove_black_masks(img, mask, save_dir = None, visualisation = False):
    if visualisation == True:
        before_img = list(img)
        before_mask = list(mask)
    idxs1 = []
    idxs2 = []
    for n in range(len(img)):
        for i in range(mask[n].shape[0]):
            if np.max(mask[n][i]) == 0:
                idx1 = i

            else: break
        img[n] = img[n][idx1 + 1:,...]
        mask[n] = mask[n][idx1 + 1:,...]
        idxs1.append(idx1 + 1)

    for n in range(len(img)):
        for i in range(mask[n].shape[0] -1, -1, -1):
            if np.max(mask[n][i]) == 0:
                idx2 = i

            else: break
        img[n] = img[n][:idx2, ...]
        mask[n] = mask[n][:idx2, ...]
        idxs2.append(idx2)

    if visualisation == True:
        save_datavisualisation2(before_img, img, save_dir + '/visualisation/remove_black_mask/', index_first= True, normalized= True, idx1 = idxs1, idx2= idxs2)
        save_datavisualisation2(before_mask, mask, save_dir + '/visualisation/remove_black_mask/', index_first=True, normalized=True, idx1 = idxs1, idx2= idxs2)


    return img, mask


def remove_black_columns(img, save_dir= None, visualisation = False):
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

    new_img = img[:,:,id1:id2]

    if visualisation:
        save_datavisualisation2(new_img, img, save_dir + '/visualisation/remove_black_columns/' , normalized= True, index_first= True)

    return new_img, id1, id2



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

def preprocess(img, shape,slice_view, save_dir = None, visualisation = False, switched_axis = False):
    """
    - moves axis such that (x,y,z) becomes (z,x,y)
    - transforms the image such that shape is (z,shape). If one dimension is bigger than shape -> downscale, if one dimension is smaller -> zero-pad
    - normalizes the data
    :param img: img with shape (x,y,z)
    :return: img with shape (z,shape)
    """
    if switched_axis == False:
        if slice_view == 'coronal':
            img = np.moveaxis(img, 1, 0)
        elif slice_view == 'transverse':
            img = np.moveaxis(img, 2, 0)

    img_data = pad_img(img, shape, save_dir ,visualisation)
    img_data = data_normalization(img_data)

    return img_data

def resample_bidsdata(path):
    """
    Resamples all the bidsdata and stores it to path
    AND changes dimensions to RAS
    """
    #fslhd header aufrufen

    if not os.path.exists(path):
        os.makedirs(path)

    bids_datas, file_names = dl.load_bidsdata()

    for i in range(len(bids_datas)):
        input_image = bids_datas[i]
        file_name = file_names[i]
        resample_cmd = 'ResampleImage 3 {input} '.format(input=input_image) + path + '{output} 0.2x0.2x0.2'.format(output=file_name)
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
    data = data*1.
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
        plt.savefig(os.path.join(path, 'img_{}.png'.format(j)))

def save_datavisualisation1(img_data, save_folder, index_first = True, normalized = False, file_names = False, file_name_header = False):
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
            i_patch = i_patch*255

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
                while os.path.exists(save_folder + file_name_header +file_names[counter] + '{}.png'.format(i)):
                    i += 1
                imageio.imwrite(save_folder + file_name_header +file_names[counter] + '{}.png'.format(i), image)
        counter = counter + 1


def save_datavisualisation2(img_data, myocar_labels, save_folder, file_name_header = False, index_first = True, normalized = False, file_names = False, idx1 = None, idx2 = None):

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
        if i.shape != j.shape:  #j need to be bigger than i     #todo this needs to be generalised
            i = np.pad(i, (((j.shape[0] - i.shape[0]) // 2, j.shape[0]- i.shape[0] - (j.shape[0] - i.shape[0]) // 2),
                                   ((j.shape[1] - i.shape[1]) // 2, j.shape[1] - i.shape[1] - (j.shape[1] - i.shape[1]) // 2), (0,0)), mode= 'constant', constant_values=0.5)
            if idx1 == None:
                j = np.pad(j, ((0,0),(0,0),((i.shape[2] - j.shape[2]) // 2, i.shape[2]- j.shape[2] - (i.shape[2] - j.shape[2]) // 2)), mode ='constant', constant_values=0.5)
            else:
                j = np.pad(j, ((0, 0), (0, 0), (idx1[counter], idx2[counter])), mode= 'constant', constant_values=0.5)


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
            path = check_path(save_folder, 'img' + '%d' % (counter,))
            imageio.imwrite(path + '.png', image)
        else:
            path = check_path(save_folder,  file_names[counter])
            imageio.imwrite(path + '.png', image)

        counter = counter + 1


def save_datavisualisation(images, save_folder, file_name_header = False, normalized = False, file_names = False):
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


        # if not type(list) == list:
        #     temp = list(list)
        #     img = []
        #     img.append(temp)


    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    for img in range(len(images[0])): #number of images that will be saved at the end
        counter = 0
        patches = []
        for list in range(len(images)):
            patch = images[list][img][0, :, :] * 255
            for slice in range(1, images[list][img].shape[0]):
                temp = images[list][img][slice,:,:] * 255
                patch = np.hstack((patch, temp))


            patches.append(patch)


        patch = patches[0]
        for i in range(1,len(patches)):
            patch = np.vstack((patch, patches[i]))

        image = np.vstack(patches)

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





def pad_img(img, shape, save_dir = None, visualisation = False):
    """
    Reshapes input image to shape. If input shape is bigger -> resize, if it is smaller -> zero-padd
    :param img:
    :param shape: shape in (y,x)
    :return:
    """
    if visualisation == True:
        before = []
        before.append(img)



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

    if visualisation == True:
        after = []
        after.append(padded)
        save_datavisualisation2(before, after, save_dir + '/visualisation/pad_img/', index_first= True, normalized= True)
    return padded


def write_blacklist(blacklist_dir):
    blacklist = []
    class blacklist_elem:
        def __init__(self, subject, session):
            self.subj = subject
            self.sess = session


    for file in os.listdir(blacklist_dir):
        temp = file.replace('.pdf', '')
        temp2 = temp.split('_')


        blacklist.append(blacklist_elem(temp2[0], temp2[1]))
    return blacklist


def check_path(path, filename = 'img', format = '.png'):
    if not os.path.exists(path):
        os.makedirs(path)
    i = 0
    while os.path.isfile(path + filename + '{}'.format(i) + format):
        i += 1

    return path + filename + '{}'.format(i)


