'''
Description: Other functions may be needed 
Author: GuoYi
Date: 2021-06-14 21:05:47
LastEditTime: 2021-06-27 09:11:47
LastEditors: GuoYi
'''


import torch
import astra 

import numpy as np 
import torch.nn as nn
import scipy.io as sio
import torch.nn.functional as F
from torch.autograd import Function
from scipy.ndimage import map_coordinates 

'''
***********************************************************************************************************
-------------UpSampling
***********************************************************************************************************
'''
## ----------------------------------------
def PixelIndexCal_DownSampling(length, width, lds, wds):
    length, width = int(length/lds), int(width/wds)
    ds_indices = torch.zeros(lds*wds, width*length).type(torch.LongTensor)

    for x in range(lds):
        for y in range(wds):
            k = x*width*wds+y
            for z in range(length):
                i, j = z*width, x*wds+y
                st = k+z*width*wds*lds
                ds_indices[j, i:i+width] = torch.tensor(range(st,st+width*wds, wds))

    return ds_indices.view(-1)


## ----------------------------------------
def PixelIndexCal_UpSampling(index, length, width):
    index = index.view(-1)
    _, ups_indices = index.sort(dim=0, descending=False)

    return ups_indices.view(-1)


## ----------------------------------------
class UpSamplingBlock(nn.Module):
    def __init__(self, planes=8, length=64, width=64, lups=2, wups=1):
        super(UpSamplingBlock, self).__init__()

        self.length = length*lups
        self.width = width*wups
        self.extra_channel = lups*wups
        self.channel = int(planes/self.extra_channel)
        ds_index = PixelIndexCal_DownSampling(self.length, self.width, lups, wups)
        self.ups_index = PixelIndexCal_UpSampling(ds_index, self.length, self.width).cuda()
        self.filter = nn.Conv2d(self.channel, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.ln = nn.GroupNorm(num_channels=planes, num_groups=1, affine=False)
        self.leakyrelu = nn.LeakyReLU(0.2, True)
        
    def forward(self, input):
        _, channel, length, width = input.size()
        channel = int(channel/self.extra_channel)
        output = torch.index_select(input.view(-1, channel, self.extra_channel*length*width), 2, self.ups_index)
        output = output.view(-1, channel, self.length, self.width)
        output = self.leakyrelu(self.ln(self.filter(output)))

        return output
        


'''
***********************************************************************************************************
-------------NetWork
***********************************************************************************************************
'''
## New Model Utils
## ----------------------------------------
class basic_block(nn.Module):
    def __init__(self, ch):
        super(basic_block, self).__init__()

        self.filter1 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu1 = nn.ReLU()
        self.filter2 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2 = nn.ReLU()

    def forward(self, input):
        output = self.relu1(self.filter1(input))
        output = self.filter2(output)
        output += input
        output = self.relu2(output)

        return output


## Convolution Four
## ----------------------------------------
class ResBasic(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3):
        super(ResBasic, self).__init__()
        self.four_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k_size, padding=int(k_size/2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),

            nn.Conv2d(out_channels, out_channels, kernel_size=k_size, padding=int(k_size/2)),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.four_conv(x)
        out = out + self.shortcut(x)
        return F.relu(out)



## Upscaling then double conv
## ----------------------------------------
class CatRes(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3):
        super(CatRes, self).__init__()
        self.res = ResBasic(in_channels, out_channels, k_size)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        return self.res(x)


## Output
## ----------------------------------------
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3):
        super(OutConv, self).__init__()
        self.conv_3 = nn.Conv2d(in_channels, int(in_channels/2), kernel_size=k_size, padding=int(k_size/2))
        self.conv_1 = nn.Conv2d(int(in_channels/2), out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv_1(self.conv_3(x))




'''
***********************************************************************************************************
-------------FBP
***********************************************************************************************************
'''
## voxel_backprojection
## ----------------------------------------
class voxel_backprojection(object):
    def __init__(self, geo):
        self.geo = geo

        self.vol_geom = astra.create_vol_geom(geo['nVoxelY'], geo['nVoxelX'], 
                                              -1*geo['sVoxelY']/2, geo['sVoxelY']/2, -1*geo['sVoxelX']/2, geo['sVoxelX']/2)
        self.proj_geom = astra.create_proj_geom(geo['mode'], geo['dDetecU'], geo['nDetecU'], 
                                                np.linspace(geo['start_angle'], geo['end_angle'], geo['sino_views'], False), geo['DSO'], geo['DOD'])
        self.proj_id = astra.create_projector('cuda', self.proj_geom, self.vol_geom) #line_fanflat
        
    def __call__(self, sinogram):
        sinogram = sinogram.view(self.geo['sino_views'], self.geo['nDetecU'])
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ProjectorId'] = self.proj_id
        cfg['FilterType'] = 'Ram-Lak'
        
        sinogram_id = astra.data2d.create('-sino', self.proj_geom, sinogram.cpu().numpy())
        rec_id = astra.data2d.create('-vol', self.vol_geom)

        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sinogram_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        image = astra.data2d.get(rec_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(sinogram_id)
        astra.algorithm.delete(alg_id)

        return torch.tensor(image).cuda().view(-1, 1, self.geo['nVoxelX'], self.geo['nVoxelY'])


## siddon_ray_projection
## ----------------------------------------
class siddon_ray_projection(object):
    def __init__(self, geo):
        self.geo = geo

        self.vol_geom = astra.create_vol_geom(geo['nVoxelY'], geo['nVoxelX'], 
                                              -1*geo['sVoxelY']/2, geo['sVoxelY']/2, -1*geo['sVoxelX']/2, geo['sVoxelX']/2)
        self.proj_geom = astra.create_proj_geom(geo['mode'], geo['dDetecU'], geo['nDetecU'], 
                                                np.linspace(geo['start_angle'], geo['end_angle'], geo['sino_views'], False), geo['DSO'], geo['DOD'])
        self.proj_id = astra.create_projector('cuda', self.proj_geom, self.vol_geom) #line_fanflat linear

    def __call__(self, image):
        image = image.view(self.geo['nVoxelX'], self.geo['nVoxelY'])
        sinogram_id, sinogram = astra.create_sino(image.cpu().numpy(), self.proj_id)
        astra.data2d.delete(sinogram_id)

        return torch.tensor(sinogram).cuda().view(-1, 1, self.geo['sino_views'], self.geo['nDetecU'])


## fan_backprojection
## ----------------------------------------
init = 'input init para'
geo = 'input geo'
bp = voxel_backprojection(geo)
fp = siddon_ray_projection(geo)


## ----------------------------------------
class backprojection(Function):     
    @staticmethod
    def forward(ctx, input):
        output = bp(input)
        return output

    @staticmethod
    def backward(ctx, image):
        sinogram = fp(image)
        return sinogram


class frontprojection(Function):     
    @staticmethod
    def forward(ctx, input):
        output = fp(input)
        return output

    @staticmethod
    def backward(ctx, sinogram):
        image = bp(sinogram)
        return image


class BP(nn.Module):
    def __init__(self, geo):
        super(BP, self).__init__()
        self.geo = geo
        self.register_parameter('weight', None)
        self.register_parameter('bias', None)
        self.backprojection = backprojection(self.geo)

    def forward(self, input):
        output = self.backprojection.apply(input)
        return output


class FP(nn.Module):
    def __init__(self, geo):
        super(FP, self).__init__()
        self.geo = geo
        self.register_parameter('weight', None)
        self.register_parameter('bias', None)
        self.frontprojection = frontprojection(self.geo)

    def forward(self, input):
        output = self.frontprojection.apply(input)
        return output



'''
***********************************************************************************************************
-------------Updata Sinogram
***********************************************************************************************************
'''
def LineInter(image, size, order = 3):
    ## scipy.ndimage.map_coordinates 
   new_dims = []
   for original_size, new_size in zip(image.shape, size):
      new_dims.append(np.linspace(0, original_size-1, new_size))
   coords = np.meshgrid(*new_dims, indexing='ij')
   '''
   Parameters:	
   input : ndarray
   The input array.
   coordinates : array_like
   The coordinates at which input is evaluated.
   output : ndarray or dtype, optional
   The array in which to place the output, or the dtype of the returned array.
   order : int, optional
   The order of the spline interpolation, default is 3. The order has to be in the range 0-5.
   mode : str, optional
   Points outside the boundaries of the input are filled according to the given mode (‘constant’, ‘nearest’, ‘reflect’ or ‘wrap’). Default is ‘constant’.
   cval : scalar, optional
   Value used for points outside the boundaries of the input if mode='constant'. Default is 0.0
   prefilter : bool, optional
   The parameter prefilter determines if the input is pre-filtered with spline_filter before interpolation (necessary for spline interpolation of order > 1). If False, it is assumed that the input is already filtered. Default is True.
   Returns:	
   map_coordinates : ndarray
   The result of transforming the input. The shape of the output is derived from that of coordinates by dropping the first axis.
   '''
   return map_coordinates(image, coords, order=order)


class Updata_sinogram(object):
    def __init__(self):
        pass

    def __call__(self, sinogram_sparse, sinogram_pred):
        sinogram_p = sinogram_pred[0,0,:,:]
        sinogram_s = sinogram_sparse[0,0,:,:]
        view_index = (np.linspace(0, sinogram_p.shape[0]-1, sinogram_s.shape[0])).astype(np.int32)
        # for i,index in enumerate(view_index):
        #     sinogram_p[index] = sinogram_s[i]
        sinogram_p[view_index] = sinogram_s
        return sinogram_p.unsqueeze_(0).unsqueeze_(0)
    

class UpdataSinogram(nn.Module):
    def __init__(self):
        super(UpdataSinogram, self).__init__()
        self.register_parameter('weight', None)
        self.register_parameter('bias', None)
        self.upsino = Updata_sinogram()
        
    def forward(self, sinogram_sparse, sinogram_pred):
        output = self.upsino(sinogram_sparse, sinogram_pred)
        return output



