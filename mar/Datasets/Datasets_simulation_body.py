'''
Description: Functions od datasets
Author: GuoYi
Date: 2021-06-14 20:21:35
LastEditTime: 2023-06-01 16:39:14
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
    def __init__(self, dataRootPath, folder, preTransImg, datasetName, mask_path, mask_folder):
        self.datasetName = datasetName
        self.preTransImg = preTransImg
        self.imgset = BasicData(dataRootPath, folder, datasetName)
        self.mask_path = mask_path
        self.mask_folder = mask_folder

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
        p = data['p']
        p_interp = data['p_interp']
        p_label = data['p_label']
        p_label1 = data['p_label1']

        sim = data['sim']
        interp_img = data['interp_img']
        sim_label = data['sim_label']
        sim_label1 = data['sim_label1']

        imBHC = data['imBHC']
        image = data['image']

        metal_mask_proj = data['metal_mask_proj']
        metal_proj = data['metal_proj']
        metal_trace_mask = data['metal_trace_mask']
        mask_path = data['maskPath']

        # read mask 
        maskdata = hdf5storage.loadmat(self.mask_path + self.mask_folder + '/' + mask_path[0][0][0][0])
        mask = maskdata['mask']

        # transform
        transf = self.calTransform(self.datasetName, self.preTransImg, self.fixList)

        p = transf(p).permute(1, 0)
        p_interp = transf(p_interp).permute(1, 0)
        p_label = transf(p_label).permute(1, 0)
        p_label1 = transf(p_label1).permute(1, 0)

        sim = transf(sim)
        interp_img = transf(interp_img)
        sim_label = transf(sim_label)
        sim_label1 = transf(sim_label1)

        imBHC = transf(imBHC)
        image = transf(image)
        
        metal_mask_proj = transf(metal_mask_proj).permute(1, 0)
        metal_proj = transf(metal_proj).permute(1, 0)
        metal_trace_mask = transf(metal_trace_mask).permute(1, 0)
        mask = transf(mask)

        # read mask 

        sample = {'p': p.unsqueeze_(0),
                'p_interp': p_interp.unsqueeze_(0),
                'p_label': p_label.unsqueeze_(0),
                'p_label1': p_label1.unsqueeze_(0),
                'sim': sim.unsqueeze_(0),
                'interp_img': interp_img.unsqueeze_(0),
                'sim_label': sim_label.unsqueeze_(0),
                'sim_label1': sim_label1.unsqueeze_(0),
                'imBHC': imBHC.unsqueeze_(0),
                'image': image.unsqueeze_(0),
                'metal_mask_proj': metal_mask_proj.unsqueeze_(0),
                'metal_proj': metal_proj.unsqueeze_(0),
                'metal_trace_mask': metal_trace_mask.unsqueeze_(0),
                'mask': mask.unsqueeze_(0),
                'name': fileName}
        return sample
