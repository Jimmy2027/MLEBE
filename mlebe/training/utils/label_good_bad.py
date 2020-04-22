import os
from mlebe.training import data_loader as dl
from mlebe.training.utils import general
import pandas as pd
from matplotlib import pyplot as plt
import copy
import functools, operator

data_dir = '/Users/Hendrik/Desktop/bartholin'
data_sets = ['drlfom', 'mgtdbs', 'opfvta', 'ztau', 'hendrik_nvcz', 'irsabi']
template_dir = '/Users/Hendrik/Documents/mlebe_data/mouse-brain-atlases'
shape = (128, 128)
slice_view = 'coronal'
if os.path.isdir('/Users/Hendrik/Documents/MLEBE/mlebe/Blacklist'):
    blacklist = general.write_blacklist('/Users/Hendrik/Documents/MLEBE/mlebe/Blacklist')
else:
    print('No Blacklist dir found')
# list = [blacklist[subj] for subj in blacklist['subject']]
if not os.path.isfile('/Users/Hendrik/Documents/MLEBE/mlebe/Blacklist/blacklist.csv'):
    blacklist_df = pd.DataFrame(columns=['subject', 'session', 'acquisition', 'modality', 'slice', 'label', 'study'])
else:
    blacklist_df = pd.read_csv('/Users/Hendrik/Documents/MLEBE/mlebe/Blacklist/blacklist.csv')

for data_set in data_sets:
    img_data = dl.load_img(data_dir, blacklist, studies = [data_set])
    temp = dl.load_mask(template_dir)
    mask_data = []
    for i in range(len(img_data)):
        mask_data.append(copy.deepcopy(temp[0]))
    img_data, mask_data, img_affines, img_headers, img_file_names, mask_affines, mask_headers = general.get_image_and_mask(img_data, mask_data, shape, save_dir ='', slice_view=slice_view, blacklist_bool=False)
    for scan in range(len(img_data)):
        print('subject ' + str(scan) + ' of ' + str(len(img_data)))
        file = img_file_names[scan].replace('.nii.gz', '').split('_')
        subject = file[0].split('-')[1]
        session = file[1].split('-')[1]
        acquisition = file[2].split('-')[1]
        modality = file[3]
        for slice in range(img_data[scan].shape[0]):
            temp_df = blacklist_df.loc[functools.reduce(operator.and_, (blacklist_df[item] == current for item, current in zip(['subject', 'session', 'acquisition', 'modality', 'slice', 'study'],[int(subject), session, acquisition, modality, slice, data_set]))), 'label']
            if temp_df.empty or temp_df.isna().all():
                plt.imshow(img_data[scan][slice], cmap= 'gray')
                plt.imshow(mask_data[scan][slice], alpha=0.3, cmap='Blues')
                plt.title(img_file_names[scan])
                plt.show(block = False)
                label = input('Good or bad?')
                if label in ['', 'q']:
                    label = 'good'
                if label in ['w', 'b']:
                    label = 'bad'
                if label in ['o']:
                    label = 'okay'
                plt.clf()
                plt.close()
                blacklist_df = blacklist_df.append(pd.DataFrame([[subject, session, acquisition, modality, slice, label, data_set]], columns=['subject', 'session', 'acquisition', 'modality', 'slice', 'label', 'study']))
                blacklist_df.to_csv('/Users/Hendrik/Documents/MLEBE/mlebe/Blacklist/blacklist.csv', index = False)
            else:
                print('already tested')
