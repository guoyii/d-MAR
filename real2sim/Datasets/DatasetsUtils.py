'''
Description: Function used in Datasets.py
Author: GuoYi
Date: 2021-06-14 20:21:06
LastEditTime: 2023-04-17 14:42:52
LastEditors: GuoYi
'''
import glob
import math 
import pydicom
import torch
import numpy as np
from scipy.ndimage import map_coordinates



'''
***********************************************************************************************************
Fixed image processing:
Scale2Gen,  FScale2Gen, RandomCrop, Normalize, FNormaliz
any2one, Any2One, ToTensor, CTnum2AtValue, 
***********************************************************************************************************
'''
## Scale2Gen [max, min] --> [0, 255]
## ----------------------------------------
class Scale2Gen(object):
    def __init__(self, scale_type='image'):
        if scale_type == 'image':
            self.mmin = -0.03718989986181259
            self.mmax = 6.623779530684153

        elif scale_type == 'self':
            self.mmin = None
            self.mmax = None

    def __call__(self, image):
        if self.mmin != None:
            img_min, img_max = self.mmin, self.mmax
        else:
            img_min, img_max = np.min(image), np.max(image)
        image = (image - img_min) / (img_max-img_min) * 255.0
        return image


## ----------------------------------------
class FScale2Gen(object):
    def __init__(self, scale_type='image'):
        if scale_type == 'image':
            self.mmin = -0.03718989986181259
            self.mmax = 6.623779530684153

        elif scale_type == 'self':
            self.mmin = None
            self.mmax = None

    def __call__(self, image):
        if self.mmin != None:
            img_min, img_max = self.mmin, self.mmax
        else:
            img_min, img_max = np.min(image), np.max(image)
        image = image / 255.0 * (img_max-img_min) + img_min
        return image


## Cut image randomly
## ----------------------------------------
class RandomCrop(object):
    def __init__(self, crop_size, crop_point=None):
        self.crop_size = crop_size
        if crop_point == None:
            self.crop_point = np.random.randint(self.crop_size, size=2)
        else:
            self.crop_point = crop_point

    def __call__(self, image):
        image = np.hstack((image, image))
        image = np.vstack((image, image))

        image = image[self.crop_point[0]:self.crop_point[0]+self.crop_size, self.crop_point[1]:self.crop_point[1]+self.crop_size]
        image = np.pad(image,((math.ceil((self.crop_size - image.shape[0])/2), math.floor((self.crop_size - image.shape[0])/2)),
                (math.ceil((self.crop_size - image.shape[1])/2), math.floor((self.crop_size - image.shape[1])/2))),'constant')
        return image


## Normalize
## ----------------------------------------
class Normalize(object):
    def __init__(self, normalize_type='image'):
        if normalize_type == 'image':
            self.mean = 128.0
            # self.mean = 0.009
        elif normalize_type == 'self':
            self.mean = None

    def __call__(self, image):
        if self.mean != None:
            img_mean = self.mean
        else:
            img_mean = np.mean(image)
        image = image - img_mean
        image = image / 255.0
        return image


## ----------------------------------------
class FNormalize(object):
    def __init__(self, normalize_type='image'):
        if normalize_type == 'image':
            self.mean = 128.0
            # self.mean = 0.009
        elif normalize_type == 'self':
            self.mean = None

    def __call__(self, image):
        if self.mean != None:
            img_mean = self.mean
        else:
            img_mean = np.mean(image)

        image = image * 255.0
        image = image + img_mean
        return image


## Image normalization
## ----------------------------------------
def any2one(image, image_max=None, image_min=None):
    if image_max == None or image_min == None:
        image_max = torch.max(image)
        image_min = torch.min(image)
    return (image-image_min)/(image_max-image_min), image_max, image_min


## ----------------------------------------
class Any2One(object):
    def __init__(self, normalize_type='self'):
        if normalize_type == 'image':
            self.image_max = 128.0
            self.image_min = 0
        elif normalize_type == 'self':
            self.image_max = None

    def __call__(self, image):
        if self.image_max != None:
            image_max = self.image_max
            image_min = self.image_min
        else:
            image_max = np.max(image)
            image_min = np.min(image)
        return (image-image_min)/(image_max-image_min)


## Change to torch
## ----------------------------------------
class ToTensor(object):
    '''Convert ndarrays in sample to Tensors.'''
    def __call__(self, image):
        # return torch.from_numpy(image.astype(np.int16)).type(torch.FloatTensor)
        return torch.from_numpy(image).type(torch.FloatTensor)


## CT to mu
## ----------------------------------------
class CTnum2AtValue(object):
    def __init__(self, WaterAtValue=None):
        self.WaterAtValue = WaterAtValue
        if self.WaterAtValue == None:
            self.WaterAtValue = 0.0192

    def __call__(self, image):
        image = image * self.WaterAtValue /1000.0 + self.WaterAtValue
        return image


## mu to CT
## ----------------------------------------
class AtValue2CTnum(object):
    def __init__(self, WaterAtValue=None):
        self.WaterAtValue = WaterAtValue
        if self.WaterAtValue == None:
            self.WaterAtValue = 0.0192

    def __call__(self, image):
        image = (image -self.WaterAtValue)/self.WaterAtValue *1000.0
        return image



'''
***********************************************************************************************************
Read Image:
findFiles, image_read
***********************************************************************************************************
'''
## Read Mayo Image
## ----------------------------------------
def findFiles(path): return glob.glob(path)



def find_zhujiang_class_file(kernel_list, thick_list, manu_list, kVp_list, mA_list, position_list, root_data_path, dataset_name, phase_name):
    # 判断上面列表是否为空，如果是，对应变量变成字符串'*'
    # 这个函数的作用是：根据输入的参数，找到对应的文件
    # 输入参数：
    # kernel_list：核心列表
    # thick_list：厚度列表
    # manu_list：制造商列表
    # kVp_list：kVp列表
    # mA_list：mA列表
    # position_list：位置列表
    # root_data_path：数据根目录
    # dataset_name：数据集名称
    # phase_name：数据集阶段名称
    # 输出参数：
    # all_file_list：所有文件列表
    # Author：GuoYi
    # Date：2021-04-16 16:01:41
    # LastEditTime：2021-04-16 16:02:00
    if len(kernel_list) == 0:
        kernel_list = ['*']
    if len(thick_list) == 0:
        thick_list = ['*']
    if len(manu_list) == 0:
        manu_list = ['*']
    if len(kVp_list) == 0:
        kVp_list = ['*']
    if len(mA_list) == 0:
        mA_list = ['*']
    if len(position_list) == 0:
        position_list = ['*']

    all_file_list = []
    data_path = root_data_path + dataset_name + '/' + phase_name + '/*/*/*/'
    # 循环遍历kernel_list
    for kernel in kernel_list:
        if kernel == 'SOFTTISSUE':
            kernel = 'SOFT TISSUE'
        if kernel == 'SoftTissue':
            kernel = 'Soft Tissue'
        # 循环遍历thick_list
        for thick in thick_list:
            # 循环遍历manu_list
            for manu in manu_list:
                # 循环遍历kVp_list
                for kVp in kVp_list:
                    # 循环mA_list
                    for mA in mA_list:
                        # 循环position_list
                        for position in position_list:
                            folder_name = 'K-{}_T-{}_M-{}_P-{}_Kv-{}_mA-{}'.format(kernel, thick, manu, position, kVp, mA)
                            data_list = glob.glob(data_path + folder_name + '/*.mat')
                            all_file_list.extend(data_list)

    return all_file_list


## Read Mayo Image
## ----------------------------------------
def image_read(image_path):
    image = pydicom.dcmread(image_path)
    return image.pixel_array * image.RescaleSlope + image.RescaleIntercept



'''
***********************************************************************************************************
Sinogram:
sparse_view_f, my_map_coordinates
***********************************************************************************************************
'''
## sparse transform
## ----------------------------------------
def sparse_view_f(sino_true,  view_origin=1160, view_sparse=60):
   view_index = (np.linspace(0, view_origin-1, view_sparse)).astype(np.int32)
   return sino_true[view_index, :]


## resize image
## ----------------------------------------
def my_map_coordinates(image, size, order = 3):
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
   The coordinates at which input == evaluated.
   output : ndarray or dtype, optional
   The array in which to place the output, or the dtype of the returned array.
   order : int, optional
   The order of the spline interpolation, default == 3. The order has to be in the range 0-5.
   mode : str, optional
   Points outside the boundaries of the input are filled according to the given mode (‘constant’, ‘nearest’, ‘reflect’ or ‘wrap’). Default == ‘constant’.
   cval : scalar, optional
   Value used for points outside the boundaries of the input if mode='constant'. Default == 0.0
   prefilter : bool, optional
   The parameter prefilter determines if the input == pre-filtered with spline_filter before interpolation (necessary for spline interpolation of order > 1). If False, it == assumed that the input == already filtered. Default == True.
   Returns:	
   map_coordinates : ndarray
   The result of transforming the input. The shape of the output == derived from that of coordinates by dropping the first axis.
   '''
   return map_coordinates(image, coords, order=order)



'''
***********************************************************************************************************
Training pretreatment:
Transpose, TensorFlip, flip, MayoTrans
***********************************************************************************************************
'''
## Transpose
## ----------------------------------------
class Transpose(object):
    def __call__(self, image):
        return image.transpose_(1, 0)


## Flip
## ----------------------------------------
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


## ----------------------------------------
class TensorFlip(object):
    def __init__(self, dim):
        self.dim = dim

    @classmethod
    def flip(cls, x, dim):
        xsize = x.size()
        dim = x.dim() + dim if dim < 0 else dim
        x = x.view(-1, *xsize[dim:])
        x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                        -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
        return x.view(xsize)

        
    def __call__(self, image):
        return self.flip(image, self.dim)


## ----------------------------------------
class MayoTrans(object):
    def __init__(self, WaterAtValue, trans_style='self'):
        self.WaterAtValue = WaterAtValue
        self.AtValue2CTnum = AtValue2CTnum(WaterAtValue)
        self.Scale2Gen = Scale2Gen(trans_style)
        self.Normalize = Normalize(trans_style)

    def __call__(self, image):
        image = self.AtValue2CTnum(image)
        image, img_min, img_max = self.Scale2Gen(image)
        image, img_mean = self.Normalize(image)

        a = 1000.0/((img_max-img_min)*self.WaterAtValue)
        b = -(img_min+1000.0)/(img_max-img_min)-img_mean/255.0
 
        return image, a, b


## ----------------------------------------
class SinoTrans(object):
    def __init__(self, trans_style='self'):
        self.Normalize = Normalize(trans_style)

    def __call__(self, sino):
        sino, img_mean = self.Normalize(sino)

        a = 1.0/255.0
        b = -img_mean/255.0

        return sino, a, b


## Add noise
## ----------------------------------------
def add_noise(noise_typ, image, mean=0, var=0.1):
    if noise_typ == 'gauss':
        row,col= image.shape
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = image + gauss
        return noisy
    elif noise_typ == 'poisson':
        noisy = np.random.poisson(image)
        return noisy

    
