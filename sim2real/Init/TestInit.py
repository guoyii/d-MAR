'''
Description: 
Author: GuoYi
Date: 2022-12-13 22:05:30
LastEditTime: 2022-12-30 10:15:46
LastEditors: GuoYi
'''

import torch
import sys
import numpy as np
import argparse
import inspect
from improved_diffusion.script_util import sr_create_model_and_diffusion 



def test_defaults():
    '''
    Defaults for image training.
    '''
    version='v1'
    root_path = '/home/gy/user/data/ImprovedDiffusion/model/IDDPM/v2'
    result_path = f'{root_path}/{version}'


    model_name = 'IDDPMModel'
    optimizer_name = 'IDDPMOptim'

    default = dict(
        gpu_id=0,
        version=version,
        mode='test',
        
        batch_size={'train': 2, 'val': 2, 'test': 1},
        epoch_num=100,

        use_cuda=torch.cuda.is_available(),
        num_workers=4,
 
        use_fp16=False,
        fp16_scale_growth=1e-3,

        schedule_sampler='uniform',

        is_lr_scheduler = True,
        is_shuffle = {'train': True, 'test': False, 'val': False},

        showNum = {'train': 50, 'test': 20, 'val': 20},

        data_root_path = '/mnt/kunlun/users/gy/data/datas/Mayo',
        root_path = root_path,

        model_name = model_name,
        optimizer_name = optimizer_name,

        result_path = result_path,
        loss_path = f'{result_path}/loss',
        model_path = f'{result_path}/model',
        optimizer_path = f'{result_path}/optimizer',
        log_path = f'{result_path}/log',

        train_folder = 'train',
        test_folder = 'test',
        val_folder = 'validation',
    )


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
        attention_resolutions='4,8,16',
        num_heads=4,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        dropout=0.0,
        # ** create_gaussian_diffusion
        diffusion_steps=1000,
        noise_schedule="cosine",
        use_kl=True, 
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        timestep_respacing='',
        # ** 超分辨率任务中被删除 
        image_size=128,
        sigma_small=False,
        
    )


def create_argparser():
    defaults = dict(
        Author='GuoYi',
        Date='2022.12.13',
        clip_denoised=True,
        num_samples=10000,
        use_ddim=False,
    )
    defaults.update(sr_model_and_diffusion_defaults())
    defaults.update(test_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def sr_model_and_diffusion_defaults():
    res = model_and_diffusion_defaults()
    res['large_size'] = 512
    res['small_size'] = 128
    arg_names = inspect.getfullargspec(sr_create_model_and_diffusion)[0]
    for k in res.copy().keys():
        if k not in arg_names:
            del res[k]
    return res

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



