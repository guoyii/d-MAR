'''
Description: Functions od datasets
Author: GuoYi
Date: 2021-06-14 20:21:35
LastEditTime: 2023-05-31 20:57:03
LastEditors: GuoYi
'''
import pdb
import os
import scipy.io as sio
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import hdf5storage
import random

from Datasets.DatasetsUtils import findFiles, ToTensor, find_zhujiang_class_file


# Basic datasets
# ----------------------------------------
# > This class is used to load the data from the .mat files and return the data and the file name
class BasicData(Dataset):
    def __init__(self, dataRootPath, folder, datasetName):
        self.folder = folder
        self.datasetName = datasetName

        # 如果datasetName不等于train，那么phase就是test
        if datasetName != 'train':
            phase = 'test'
        else:
            phase = 'train'
        kernel_list = folder['kernel_list']
        thick_list = folder['thick_list']
        manu_list = folder['manu_list']
        kVp_list = folder['kVp_list']
        mA_list = folder['mA_list']
        position_list = folder['position_list']
        dataset_name = folder['dataset_name']
        data_list = find_zhujiang_class_file(kernel_list, thick_list, manu_list, kVp_list, mA_list, position_list, dataRootPath, dataset_name, phase )

        # 如果datasetName等于'val', 那么就从data_list中选取前500个作为验证集
        if datasetName == 'train':
            data_list = data_list[0:]
        elif datasetName == 'val':
            data_list = data_list[0:800]
        else:
            random.shuffle(data_list)
            # pdb.set_trace()
            data_list = data_list[0:900]
        self.pathList = data_list

    def __len__(self):
        return len(self.pathList)

    def __getitem__(self, idx):
        data = hdf5storage.loadmat(self.pathList[idx])
        fileName = os.path.basename(self.pathList[idx])
        return data, fileName



# ----------------------------------------
class BuildDataSet(Dataset):
    def __init__(self, dataRootPath, folder, preTransImg, datasetName):
        self.datasetName = datasetName
        self.preTransImg = preTransImg
        self.imgset = BasicData(dataRootPath, folder, datasetName)

        self.fixList = [ToTensor()]

    def __len__(self):
        return len(self.imgset)

    @classmethod
    def calTransform(cls, datasetName, preTransImg, fixList):
        randomList = []
        if datasetName == 'train':
            if preTransImg != None:
                keys = np.random.randint(2, size=len(preTransImg))
                for i, key in enumerate(keys):
                    randomList.append(preTransImg[i]) if key == 1 else None
        transform = transforms.Compose(fixList + randomList)
        return transform


    def __getitem__(self, idx):
        data, fileName = self.imgset[idx]
        
        # %read data
        image = data['image']


        # transform
        transf = self.calTransform(self.datasetName, self.preTransImg, self.fixList)
        image = transf(image)

        sample = {'image': image.unsqueeze_(0),
                'name': fileName}
        return sample



