import numpy as np
import torch

import utils

x_test = torch.load('x_test')
y_test = torch.load('y_test')
y_pred = np.load('results/y_pred.npy')

output = []

for i in range(len(y_test)):
    output.append(np.squeeze(y_pred[i]))

utils.save_datavisualisation3(x_test, y_test, output, 'results/', index_first=True)

