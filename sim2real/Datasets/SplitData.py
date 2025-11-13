'''
Description: 
Author: GuoYi
Date: 2021-06-29 08:59:46
LastEditTime: 2021-06-29 10:02:24
LastEditors: GuoYi
'''
import os 
import pdb 
import glob 
import h5py 
import numpy as np 
import  matplotlib.pylab as plt 
import scipy.io as sio 

'''
--------------------------------------------------------
One sample
--------------------------------------------------------
'''
# inPath = '/home/gy/usr/datas/LoDoPall/test/*.hdf5'
outPath = '/home/gy/usr/datas/LoDoPall/test1/'
# pathList = glob.glob(inPath)
# print(pathList)

# # --------------------------------------------------------------
# f = h5py.File(pathList[0], 'r')
# sino_data      = np.array(f['data'], dtype='float32')
# sino_data_true = np.array(f['data_true'], dtype='float32')
# img_data       = np.array(f['data_img'], dtype='float32')
# img_data_true  = np.array(f['data_true_img'], dtype='float32')
# f.close()

# for i in range(sino_data.shape[0]):
#     sio.savemat(outPath + 'high_{}.mat'.format(i), {'sino':sino_data_true[i], 'image':img_data_true[i]})
#     sio.savemat(outPath + 'low_{}.mat'.format(i), {'sino':sino_data[i], 'image':img_data[i]})

# # --------------------------------------------------------------

# data_load = sio.loadmat(outPath + 'high_100.mat')
# image_load = data_load['image']
# sino_load = data_load['sino']

# data_load = sio.loadmat(outPath + 'low_100.mat')
# image_load1 = data_load['image']
# sino_load1 = data_load['sino']

# # pdb.set_trace()
# plt.imshow(image_load, 'gray')
# plt.subplot(121), plt.imshow(image_load, 'gray')
# plt.subplot(122), plt.imshow(image_load1, 'gray'), plt.show()


# plt.subplot(121), plt.imshow(sino_load, 'gray')
# plt.subplot(122), plt.imshow(sino_load1, 'gray'), plt.show()


'''
--------------------------------------------------------
Batch
--------------------------------------------------------
'''
inPath = '/home/gy/usr/datas/LoDoPall/'
rootOutPath = '/home/gy/usr/datas/LoDoPall/'

folders = ['observation_train', 'observation_validation', 'observation_test']
outFolders = ['simulation_train', 'simulation_validation', 'simulation_test']


for folderIndex in range(len(folders)):
    print(folders[folderIndex])
    inputPathList = glob.glob(inPath + folders[folderIndex] + '/*.hdf5')

    highOutPath = rootOutPath + outFolders[folderIndex] + '/high/'
    lowOutPath = rootOutPath + outFolders[folderIndex] + '/low/'
    if not os.path.exists(highOutPath):
        try:
            os.mkdir(highOutPath)
        except:
            os.makedirs(highOutPath)

    if not os.path.exists(lowOutPath):
        try:
            os.mkdir(lowOutPath)
        except:
            os.makedirs(lowOutPath)


    print(len(inputPathList))
    for dataIndex in range(len(inputPathList)):
        f = h5py.File(inputPathList[dataIndex], 'r')
        sino_data      = np.array(f['data'], dtype='float32')
        sino_data_true = np.array(f['data_true'], dtype='float32')
        img_data       = np.array(f['data_img'], dtype='float32')
        img_data_true  = np.array(f['data_true_img'], dtype='float32')
        f.close()
        
        
        print('Process:' + folders[folderIndex] + '--{}'.format(dataIndex))
        for sliceIndex in range(img_data.shape[0]):
            sio.savemat(highOutPath + '/data{}_slice{}.mat'.format(dataIndex, sliceIndex), {'sino':sino_data_true[sliceIndex], 'image':img_data_true[sliceIndex]})
            # sio.savemat(lowOutPath + '/data{}_slice{}.mat'.format(dataIndex, sliceIndex), {'sino':sino_data[sliceIndex], 'image':img_data[sliceIndex]})
        

# # --------------------------------------------------------------


# high = sio.loadmat('/home/gy/usr/datas/LoDoPall/simulation_train/high/data0_slice4.mat')
# image_high = high['image']
# sino_high = high['sino']

# low = sio.loadmat('/home/gy/usr/datas/LoDoPall/simulation_train/low/data0_slice4.mat')
# image_low = low['image']
# sino_low = low['sino']

# pdb.set_trace()
# plt.subplot(131), plt.imshow(image_high, 'gray')
# plt.subplot(132), plt.imshow(image_low, 'gray')
# plt.subplot(133), plt.imshow(image_high-image_low, 'gray'), plt.show()

# plt.subplot(131), plt.imshow(sino_high, 'gray')
# plt.subplot(132), plt.imshow(sino_low, 'gray')
# plt.subplot(133), plt.imshow(sino_high-sino_low, 'gray'), plt.show()


print('Run Done')