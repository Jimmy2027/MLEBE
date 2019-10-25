import data_loader as dl
import os

bids_datas, file_names = dl.load_bidsdata()
path = '/var/tmp/results/'
if not os.path.exists(path):
    os.makedirs(path)

for i in range(len(bids_datas)):
    input_image = bids_datas[i]
    cmd = 'ResampleImage 3 {input} '.format(input=input_image) + path + '{output} 0.2x0.2x0.2'.format(output = file_names)
    os.system(cmd)


