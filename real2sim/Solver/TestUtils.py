'''
Description: Functions for test.py 
Author: GuoYi
Date: 2021-06-14 22:24:11
LastEditTime: 2021-06-14 22:24:43
LastEditors: GuoYi
'''

import numpy as np 
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


## show ssim mse psnr
##******************************************************************************************************************************
def ssim_mse_psnr(image_true, image_test):
    mse = compare_mse(image_true, image_test)
    ssim = compare_ssim(image_true, image_test)
    psnr = compare_psnr(image_true, image_test, data_range=255)
    return ssim, mse, psnr

    