'''
Description: Train the model 
Author: GuoYi
Date: 2021-06-14 21:22:23
LastEditTime: 2023-06-25 00:22:33
LastEditors: GuoYi
'''
import os
import pdb 
import sys 
import time
import copy 
import tqdm 
import math
import numpy as np
import scipy.io as sio
from scipy.io import loadmat 
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import matplotlib.pylab as plt
from torch.optim import AdamW, Adam
import functools
from copy import deepcopy 


from Solver.TrainUtils import weightsInit, modelUpdata, optimizerUpdata, modelUpdataSubNet
from improved_diffusion.fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from improved_diffusion.nn import update_ema
from improved_diffusion.resample import LossAwareSampler, UniformSampler

INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        model,
        diffusion,
        schedule_sampler, 
        dataloaders,
        dataLength,
        writerLog,
        args,
    ):
        self.model = model
        self.diffusion = diffusion
        self.schedule_sampler = schedule_sampler
        self.dataloaders = dataloaders
        self.dataLength = dataLength
        self.writerLog = writerLog
        self.config = args 
        self.ema_model = EmaModel(model, 0.999)

        self.batch_size = self.config.batch_size
        self.lr = self.config.lr


        self.use_fp16 = self.config.use_fp16
        self.fp16_scale_growth = self.config.fp16_scale_growth

        self.schedule_sampler = self.schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = self.config.weight_decay
        self.lr_anneal_steps = self.config.lr_anneal_steps

        self.step = 0

        # self.model_params = list(self.model.parameters())
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE

        if self.use_fp16:
            self._setup_fp16()
        
        # self.opt = AdamW(self.model_params, lr=self.lr[0], weight_decay=self.weight_decay) 
        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr[0], betas=(0.9, 0.999))

        if self.config.re_load:
            print('Re_load is True !')
            print('Please set the path of expected model!')
            self.startEpoch = self.config.startEpoch + 1
            time.sleep(3)
            self._load_model_parameters()
            self._load_optimizer_state() 
            # self._load_ema_parameters()

            losses1 = loadmat(self.config.loss_path + '/losses.mat')
            self.lossesResult = {x: torch.zeros(self.config.epoch_num, 2) for x in ['train', 'val']}
            self.lossesResult['train'] = torch.from_numpy(losses1['train'])
            self.lossesResult['val'] = torch.from_numpy(losses1['val'])
            
            self.minLoss = {x:10000 for x in ['train', 'val']} 
            self.minLossEpoch = {x:-1 for x in ['train', 'val']} 
        else:
            self.startEpoch = 0
            self.lossesResult = {x: torch.zeros(self.config.epoch_num, 2) for x in ['train', 'val']}
            self.minLoss = {x:10000 for x in ['train', 'val']}
            self.minLossEpoch = {x:-1 for x in ['train', 'val']} 
            try:
                print('load trained model...')
                fullModelPath = '/home/gy/user/data/MAR/DDMAR/v1/v1/model/DDMAR_E85.pkl'
                fullOptimPath = ''
                # self._load_model_parameters(fullModelPath)
                # self._load_ema_parameters(fullModelPath)
                # self._load_optimizer_state(fullOptimPath) 
            except:
                print('No update...')
        self.show_batch_num = {x:int(dataLength[x]/self.config.showNum[x]/self.config.batch_size[x]) for x in ['train', 'val']}
    

      
    def _setup_fp16(self):
        self.model_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()


    def run_loop(self):
        timeAllStart = time.time()
        for epoch in tqdm.tqdm(range(self.startEpoch, self.config.epoch_num)):
            timeEpochStart = time.time()
            print('-' * 60)
            print('.........Epoch: {}/{}..........'.format(epoch, self.config.epoch_num))
            print('-' * 60)
            mu = 300/1000*0.206 + 0.206
            for phase in ['train', 'val']:
                print('\n=========== Now, Start {}==========='.format(phase))
                # pdb.set_trace()
                if phase == 'train':
                    self.model.train()
                elif phase == 'val':
                    self.model.eval()
                
                epochLoss = 0
                for i, batch in enumerate(self.dataloaders[phase]):
                    if i > 15000:
                        break  
                    timeBatchStart = time.time()
                    image = batch['sim']
                    image = Variable(image).cuda()

                    mask = image > mu 
                    image[mask] = mu

                    image = (image - 0)/(mu - 0)

                    x = image
                    cond = copy.deepcopy(image)


                    self.run_step(x, cond, phase)

                    epochLoss += self.loss.item()*image.size(0)  
                    if i>0 and math.fmod(i, self.show_batch_num[phase]) == 0:
                        showlr = self.opt.state_dict()['param_groups'][0]['lr'] 
                        print(f'{self.config.model_name} Epoch:{epoch} Batch:{i-self.show_batch_num[phase]}-{i} Learning Rate:{showlr:.6}  **{phase}**  Loss:{self.loss.item():.6f} Time:{(time.time()-timeBatchStart)*self.show_batch_num[phase]:.4f}s')

              
                self.writerLog.add_scalars('Loss', {phase: epochLoss/self.dataLength[phase]}, global_step=epoch)
                self.writerLog.add_scalar(phase + 'Loss', epochLoss/self.dataLength[phase], global_step=epoch)

                # pdb.set_trace()

                epochLoss = epochLoss/self.dataLength[phase]
                self.lossesResult[phase][epoch, 1] = epochLoss
                if epochLoss < self.minLoss[phase]:
                    self.save_best_model(phase, epochLoss, epoch)
                print('Epoch {} Average {} loss:{:.8f}'.format(epoch, phase, epochLoss))

            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = self.config.lr*pow(self.config.gamma, int(epoch/self.config.step_size))
            for param_group in self.opt.param_groups:
                param_group['lr'] = self.config.lr[min(epoch+1, len(self.config.lr)-1)]
            
            self.save_model(epoch)
            print('Time for epoch {} : {:.4f}min'.format(epoch+1, (time.time()-timeEpochStart)/60))
            print('Time for ALL : {:.4f}h\n'.format((time.time()-timeAllStart)/3600))   

        ## ********************************************************************************************************************
        print('\nTrain Completed!! Time for ALL : {:.4f}h'.format((time.time()-timeAllStart)/3600)) 


    def run_step(self, batch, cond, phase):
        self.forward_backward(batch, cond, phase)


    def forward_backward(self, batch, cond, phase):
        # zero_grad(self.model_params)
        self.opt.zero_grad()
        micro = batch
        micro_cond = {'origin_img': cond}
        
        
        t, weights = self.schedule_sampler.sample(micro.shape[0], 'cuda')

        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.model,
            micro,
            t,
            model_kwargs=micro_cond,
        )

        losses = compute_losses()


        self.loss = (losses['loss'] * weights).mean()
        
        # self.loss = losses['loss'].mean() 


        if phase == 'train':
            self.loss.backward()
            self.opt.step() 
            # self.ema_model.update(self.model)
        


    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            return

        model_grads_to_master_grads(self.model_params, self.model_params)
        self.model_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.model_params, rate=rate)
        master_params_to_model_params(self.model_params, self.model_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.model_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.model_params:
            sqsum += (p.grad ** 2).sum().item()

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = self.step  / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model.parameters(), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict


    def save_best_model(self, phase, epochLoss, epoch):
        temp = self.minLossEpoch[phase]
        self.minLoss[phase] = epochLoss
        self.minLossEpoch[phase]= epoch

        data_save = {key: value for key, value in self.minLoss.items()}
        sio.savemat(self.config.loss_path + '/minLoss.mat', mdict = data_save)

        data_save = {key: value for key, value in self.minLossEpoch.items()}
        sio.savemat(self.config.loss_path + '/minLossEpoch.mat', mdict = data_save)

        torch.save(self.model.state_dict(), self.config.model_path + '/' + self.config.model_name + '_E{}_{}_Best.pkl'.format(epoch, phase))
        # torch.save(self.ema_model.state_dict(), self.config.model_path + '/' + self.config.model_name + '_E{}_{}_Best_ema.pkl'.format(epoch, phase))
        torch.save(self.opt.state_dict(), self.config.optimizer_path + '/' + self.config.optimizer_name + '_E{}_{}_Best.pkl'.format(epoch, phase))

        if os.path.exists(self.config.model_path + '/' + self.config.model_name + '_E{}_{}_Best.pkl'.format(temp, phase)):
            os.unlink(self.config.model_path + '/' + self.config.model_name + '_E{}_{}_Best.pkl'.format(temp, phase))
        # if os.path.exists(self.config.model_path + '/' + self.config.model_name + '_E{}_{}_Best_ema.pkl'.format(temp, phase)):
        #     os.unlink(self.config.model_path + '/' + self.config.model_name + '_E{}_{}_Best_ema.pkl'.format(temp, phase))
        if os.path.exists(self.config.optimizer_path + '/' + self.config.optimizer_name + '_E{}_{}_Best.pkl'.format(temp, phase)):
            os.unlink(self.config.optimizer_path + '/' + self.config.optimizer_name + '_E{}_{}_Best.pkl'.format(temp, phase))
        

    
    def save_model(self, epoch):
        torch.save(self.model.state_dict(), self.config.model_path + '/' + self.config.model_name + '_E{}.pkl'.format(epoch))
        # torch.save(self.ema_model.state_dict(), self.config.model_path + '/' + self.config.model_name + '_E{}_ema.pkl'.format(epoch))
        torch.save(self.opt.state_dict(), self.config.optimizer_path + '/' + self.config.optimizer_name + '_E{}.pkl'.format(epoch))

        if os.path.exists(self.config.model_path + '/' + self.config.model_name + '_E{}.pkl'.format(epoch-1)):
            os.unlink(self.config.model_path + '/' + self.config.model_name + '_E{}.pkl'.format(epoch-1))
        # if os.path.exists(self.config.model_path + '/' + self.config.model_name + '_E{}_ema.pkl'.format(epoch-1)):
        #     os.unlink(self.config.model_path + '/' + self.config.model_name + '_E{}_ema.pkl'.format(epoch-1))
        if os.path.exists(self.config.optimizer_path + '/' + self.config.optimizer_name + '_E{}.pkl'.format(epoch-1)):
            os.unlink(self.config.optimizer_path + '/' + self.config.optimizer_name + '_E{}.pkl'.format(epoch-1))

        if epoch>=self.config.epoch_num-1:
            torch.save(self.model.state_dict(), self.config.model_path + '/' + self.config.model_name + '_E{}.pkl'.format(epoch))
            # torch.save(self.ema_model.state_dict(), self.config.model_path + '/' + self.config.model_name + '_E{}_ema.pkl'.format(epoch))
            torch.save(self.opt.state_dict(), self.config.optimizer_path + '/' + self.config.optimizer_name + '_E{}.pkl'.format(epoch))

        data_save = {key: value.cpu().squeeze_().data.numpy() for key, value in self.lossesResult.items()}
        sio.savemat(self.config.loss_path + '/losses.mat', mdict = data_save)


    def _load_model_parameters(self, modelReloadPath=False):
        if not modelReloadPath:
            modelReloadPath = self.config.old_model_path + '/' + self.config.old_model_name + '.pkl'
        print('Old model path: {}'.format(modelReloadPath))
        if os.path.isfile(modelReloadPath):
            print('Loading previously trained network...')
            checkpoint = torch.load(modelReloadPath, map_location = lambda storage, loc: storage)
            model_dict = self.model.state_dict()
            checkpoint =  {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(checkpoint)
            self.model.load_state_dict(model_dict)
            del checkpoint
            torch.cuda.empty_cache()
            print('Loading Done!\n')
        else:
            print('\nLoading Fail!\n')
            sys.exit(0)


    def _load_ema_parameters(self, modelReloadPath=False):
        if not modelReloadPath:
            modelReloadPath = self.config.old_model_path + '/' + self.config.old_model_name + '_ema.pkl'
        print('Old model path: {}'.format(modelReloadPath))
        if os.path.isfile(modelReloadPath):
            print('Loading previously trained network...')
            checkpoint = torch.load(modelReloadPath, map_location = lambda storage, loc: storage)
            model_dict = self.model.state_dict()
            checkpoint =  {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(checkpoint)
            self.model.load_state_dict(model_dict)
            del checkpoint
            torch.cuda.empty_cache()
            print('Loading Done!\n')
        else:
            print('\nLoading Fail!\n')
            sys.exit(0)


    def _load_optimizer_state(self, optimizerReloadPath=False):
        if not optimizerReloadPath:
            optimizerReloadPath = self.config.old_optimizer_path + '/' + self.config.old_optimizer_name + '.pkl'
        print('Old model path: {}'.format(optimizerReloadPath))
        if os.path.isfile(optimizerReloadPath):
            print('Loading previously trained network...')
            checkpoint = torch.load(optimizerReloadPath, map_location = lambda storage, loc: storage)
            optim_dict = self.opt.state_dict()
            checkpoint =  {k: v for k, v in checkpoint.items() if k in optim_dict}
            optim_dict.update(checkpoint)
            self.opt.load_state_dict(optim_dict)
            del checkpoint
            torch.cuda.empty_cache()
            print('Loading Done!\n')
        else:
            print('\nLoading Fail!\n')
            sys.exit(0)
    

    def save_ema(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            filePath = self.config.ema_path + '/' + self.config.ema_name + f'_rate_{rate}.pkl'
            torch.save(state_dict, filePath)

        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)


    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(list(self.model.parameters()))

        ema_checkpoint = self.config.ema_path + '/' + self.config.ema_name + f'_rate_{rate}.pkl'
        if ema_checkpoint:
            checkpoint = torch.load(ema_checkpoint, map_location = lambda storage, loc: storage)
            model = copy.deepcopy(self.model)
            model_dict = model.state_dict()
            checkpoint =  {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(checkpoint)
            model.load_state_dict(model_dict)

            ema_params = list(self.model.parameters())

        return ema_params


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)


class EmaModel(nn.Module):
    def __init__(self, model, alpha=0.999):
        super(EmaModel, self).__init__()
        self.model: nn.Module = deepcopy(model)
        self.alpha = alpha
    
    def update(self, model):
        assert type(self.model) == type(model)
        for n, p in self.model.named_parameters():
            p.data = model.get_parameter(n).data * self.alpha + p.data * (1 - self.alpha)
            
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)



