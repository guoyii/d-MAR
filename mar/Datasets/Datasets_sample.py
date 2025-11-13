'''
Description: Functions od datasets
Author: GuoYi
Date: 2021-06-14 20:21:35
LastEditTime: 2023-07-28 11:44:57
LastEditors: GuoYi
'''
import pdb
import os
import scipy.io as sio
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import hdf5storage

from Datasets.DatasetsUtils import findFiles, ToTensor


# Basic datasets
# ----------------------------------------
# > This class is used to load the data from the .mat files and return the data and the file name
class BasicData(Dataset):
    def __init__(self, dataRootPath, folder, datasetName):
        self.folder = folder
        self.datasetName = datasetName
        self.pathList = findFiles(dataRootPath + folder + '/*.mat')

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
        # sio.savemat(outPath + name, {'image': sample[0, 0, :, :].cpu().numpy(), 'ref': image[0, 0, :, :].cpu().numpy(), 'label':data['image'],
        # 'mask': metal[0, 0, :, :].cpu().numpy()})

        sim = data['image']
        sim_label = data['label']
        mask = data['mask']


        # transform
        transf = self.calTransform(self.datasetName, self.preTransImg, self.fixList)

        sim = transf(sim)
        sim_label = transf(sim_label)
        mask = transf(mask)

        # read mask 

        sample = {
                'sim': sim.unsqueeze_(0),
                'sim_label': sim_label.unsqueeze_(0),
                'mask': mask.unsqueeze_(0),
                'name': fileName}
        return sample

