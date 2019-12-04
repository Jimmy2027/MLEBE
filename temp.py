hihimask = []
for i in range(img_temp[...,0].shape[1]):
    if max(img_temp[:,i,0]) > 0:
        hihimask.append(img_temp[:,i,0])

something = np.array(hihimask)
something = np.swapaxes(something, 0,1)