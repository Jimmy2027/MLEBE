from matplotlib import pyplot as plt
import numpy as np
import skimage
from skimage.morphology import *

import skimage.morphology.watershed
from skimage import filters
from scipy import ndimage
# image = plt.imread('psi_slice_noisy.tif')
# #
# # def maskedOtsu(img,mask) :
# #     data=[]
# #     dims=img.shape
# #     for r in np.arange(0,dims[1]) :
# #         for c in np.arange(0,dims[0]) :
# #             if (mask[r,c]!=0) :
# #                 data.append(img[r,c])
# #     ndata=np.asarray(data)
# #     return filters.threshold_otsu(ndata)
# # fimg= ndimage.filters.gaussian_filter(image[...,0], [2,2])
# # otsu = filters.threshold_otsu(fimg)
# # bimg = fimg >= otsu
# # otsu2=maskedOtsu(fimg,bimg)
# # print('The threshold according to',otsu2)
# # bimg2= fimg >= otsu2 # apply the masked threshold on fimg here plt.figure(figsize=[15,15])
# # plt.imshow(bimg2)

x, y = np.indices((80, 80))
x1, y1, x2, y2, x3, y3= 20, 20, 44, 52, 10, 50
r1, r2, r3 = 10, 15, 5
mask_circle1 = (x - x1)**2 + (y - y1)**2 < r1**2
mask_circle2 = (x - x2)**2 + (y - y2)**2 < r2**2
mask_circle3 = (x - x3)**2 + (y - y3)**2 < r3**2
image = np.logical_or(mask_circle1, mask_circle2, mask_circle3)

something = label(image)
unique, counts = np.unique(something, return_counts= True)
print(unique, counts)
plt.imshow(something)
plt.show()
# distance = ndimage.distance_transform_edt(image)
# plt.imshow(distance)
# plt.show()
# markers = ndimage.label(image)[0]
# plt.imshow(markers)
# plt.show()
# labels = watershed(-distance, markers, mask=image)
# print(len(np.unique(labels)))
# print(np.count(np.unique(labels)))
# new = np.where(labels == np.max(labels), 1, 0)
# plt.imshow(new)
# plt.show()
