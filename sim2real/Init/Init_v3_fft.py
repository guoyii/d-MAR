'''
 Description  : 
 Author       : Guo Yi
 Date         : 2022-12-13 16:50:05
LastEditors: GuoYi
LastEditTime: 2023-06-14 22:54:13
 FilePath     : \\v1\\Init\\Init.py
 Copyright (C) 2022 Guo Yi. All rights reserved.
'''


import torch
import sys
import copy 
import numpy as np
import argparse
import inspect
from improved_diffusion.script_util_v3_fft import create_model_and_diffusion 


def train_defaults():
    '''
    Defaults for image training.
    '''
    version='v3'
    root_path = '/mnt/kunlun/users/gy/data/MAR/MAR2023/TTAMAR/TTAMAR/v1'
    result_path = f'{root_path}/{version}'

    re_load_flag = True    
    model_name = 'TTAMAR'
    optimizer_name = 'TTAOptim'

    epoch_num=100

    default = dict(
        gpu_id=1,
        version=copy.deepcopy(version),
        mode='train',
        re_load=re_load_flag,
        
        batch_size={'train': 2, 'val': 1, 'test': 1},
        epoch_num=epoch_num,

        use_cuda=torch.cuda.is_available(),
        num_workers=2,

        lr = np.linspace(1e-4, 1e-6, epoch_num),  # ** 
        weight_decay=0.0,
        lr_anneal_steps=0,


        use_fp16=False,
        fp16_scale_growth=1e-3,

        schedule_sampler='uniform',

        is_lr_scheduler = True,
        is_shuffle = {'train': True, 'test': False, 'val': False},

        showNum = {'train': 50, 'test': 20, 'val': 20},

        data_root_path = '/mnt/kunlun/users/gy/data/datas/MAR/Zhujiang_class/',
        root_path = copy.deepcopy(root_path),

        model_name = model_name,
        clip_denoised = True, 
        optimizer_name = optimizer_name,

        result_path = result_path,
        loss_path = f'{result_path}/loss',
        model_path = f'{result_path}/model',
        optimizer_path = f'{result_path}/optimizer',
        log_path = f'{result_path}/log',

        train_folder = {'kernel_list':[], 'thick_list':[], 'manu_list':[], 'kVp_list':[], 'mA_list':[], 'position_list':['Head'], 'dataset_name':'normal'},
        test_folder = {'kernel_list':[], 'thick_list':[], 'manu_list':[], 'kVp_list':[], 'mA_list':[], 'position_list':['Head'], 'dataset_name':'normal'},
        val_folder = {'kernel_list':[], 'thick_list':[], 'manu_list':[], 'kVp_list':[], 'mA_list':[], 'position_list':['Head'], 'dataset_name':'normal'}
    )
    
    if re_load_flag:
        old_version = 'v3'
        old_result_path = f'{root_path}/{old_version}'
        startEpoch = 80 
        
        re_arg = dict(
            old_version = old_version,
            startEpoch = startEpoch,
            old_result_path = old_result_path,
            old_model_path = f'{old_result_path}/model',
            old_optimizer_path = f'{old_result_path}/optimizer',
            old_model_name = model_name + '_E' + str(startEpoch),
            old_optimizer_name = optimizer_name + '_E' + str(startEpoch),
        )
        default.update(re_arg)
    return default


def model_and_diffusion_defaults():
    '''
    Defaults for image training.
    '''
    return dict(
        # ** U-Net 
        num_channels=64,
        num_res_blocks=2,
        learn_sigma=False,
        class_cond=False,
        use_checkpoint=False,
        attention_resolutions='16, 8, 4',   
        num_heads=4,
        num_heads_upsample=-1, 
        use_scale_shift_norm=True,
        dropout=0.0,
        # ** create_gaussian_diffusion
        diffusion_steps=1000,
        noise_schedule="cosine",
        use_kl=False, 
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        timestep_respacing='',
    )


def mar_model_and_diffusion_defaults():
    res = model_and_diffusion_defaults()
    res['image_size'] = 512
    arg_names = inspect.getfullargspec(create_model_and_diffusion)[0]
    for k in res.copy().keys():
        if k not in arg_names:
            del res[k]
    return res


def create_argparser():
    defaults = dict(
        Author='GuoYi',
        Date='2023.5.31'
    )
    defaults.update(mar_model_and_diffusion_defaults())
    defaults.update(train_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f'--{k}', default=v, type=v_type)


def str2bool(v):
    '''
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    '''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('boolean value expected')

