import numpy as np
import pandas as pd
import os
import cv2
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

### Extract file paths from MLTrainingImages
mydir = 'C:/Users/381792/Documents/MLTrainingImages'
fpath = {}
i = 0
for root, dirs, files in os.walk(mydir):
    for file in files:
        if file.endswith('.npy'):
            fpath[i] = os.path.join(root,file)
            i += 1
filepath = list(fpath.values())

### Load facies realizations
facies = np.zeros((318,256*256*128))
for i in range(len(filepath)):
    facies[i] = np.load(filepath[i]).squeeze()
temp = facies.reshape(318,256,256,128)
print(temp.shape)

### Select 4 slices 
temp0 = temp[:,:,:,48]
temp1 = temp[:,:,:,64]
temp2 = temp[:,:,:,80]
temp3 = temp[:,:,:,96]
newtemp = np.concatenate([temp0, temp1, temp2, temp3])
print(newtemp.shape)

### Resize slices from (256,256) to (128,128)
temp_facies = np.zeros((1272,128,128))
for i in range(1272):
    temp_facies[i] = cv2.resize(newtemp[i], (128,128))
print(temp_facies.shape)

### Scale realizations to [0.8,1.5]
newtemp_facies = np.zeros((1272,128,128))
for i in range(1272):
    newtemp_facies[i] = MinMaxScaler((0.8,1.5)).fit_transform(temp_facies[i].reshape(128*128).reshape(-1,1)).reshape(128,128)
newtemp_facies.shape

### Save realizations as .mat
facies = newtemp_facies[:1000].reshape(1000,128*128)
facies.shape
for i in range(1000):
    mdict = {'facies':facies[i]}
    savemat('facies/facies{}.mat'.format(i+1), mdict)
    
### Generate conditioning data for SGeMS
idx, idy = np.random.randint(0, 127, 100), np.random.randint(0, 127, 100)
idz = np.zeros(100, dtype='int')
vals = facies[754].reshape(128,128)[idx,idy] + np.abs(np.random.normal(0,0.5,100))
print(idx.shape, idy.shape, vals.shape)

hard_data = pd.DataFrame({'X':idx, 'Y':idy, 'z':idz, 'k':vals})
print('min = {:.3f}  | max = {:.3f}'.format(10**hard_data.min()[-1], 10**hard_data.max()[-1]))
hard_data.to_csv('sgsim_hard_data.txt', sep=' ', index=None)


### Visualize realizations
plt.figure(figsize=(30,15))
for i in range(90):
    plt.subplot(9,10,i+1)
    plt.imshow(facies[i*5].reshape(128,128), 'jet')
    plt.xticks([]); plt.yticks([])
plt.savefig('facies_realizations.png')
plt.show()