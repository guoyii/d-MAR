'''
Description: Test the model
Author: GuoYi
Date: 2021-06-14 22:23:20
LastEditTime: 2021-06-27 09:48:39
LastEditors: GuoYi
'''


import os
import time
import math
import numpy as np
import scipy.io as sio
from scipy.io import loadmat 
import torch
from torch.autograd import Variable

from Solver.TestUtils import ssim_mse_psnr 

def testModel(model,
              dataloaders,
              args):
        model.eval()
        print('**************  Test  ****************')
        for i, batch in enumerate(dataloaders['test']):
            print('Now testing {} sample......'.format(i))
            image = batch['image']
            sino = batch['sino']

            if args.use_cuda:
                sino = Variable(sino).cuda()
                image = Variable(image).cuda()
            else:
                image = Variable(image)
                sino = Variable(sino)

            with torch.no_grad():
                sino_pred = model(sino)
        

