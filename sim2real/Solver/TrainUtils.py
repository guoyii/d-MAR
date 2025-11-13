'''
Description: Functions for train.py 
Author: GuoYi
Date: 2021-06-14 21:34:17
LastEditTime: 2021-06-15 17:00:39
LastEditors: GuoYi
'''
import os 
import sys 
import torch 
import torch.nn.init as init

## Init the model
## ----------------------------------------
def weightsInit(m):
     classname = m.__class__.__name__
     if classname.find('Conv2d') != -1:
        # init.normal_(m.weight.data, mean=0, std=0.001)
        init.normal_(m.weight.data, mean=0, std=1)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
        # print('Init {} Parameters.................'.format(classname))
     elif classname.find('ConvTranspose2d') != -1:
        init.normal_(m.weight.data, mean=0, std=1)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
        # print('Init {} Parameters.................'.format(classname))
     else:
        # print('{} Parameters Do Not Need Init !!'.format(classname))
        pass 


## Load model
## ----------------------------------------
def modelUpdata(model, modelOldName, modelOldPath):
    modelReloadPath = modelOldPath + '/' + modelOldName + '.pkl'
    print('\nOld model path：{}'.format(modelReloadPath))
    if os.path.isfile(modelReloadPath):
        print('Loading previously trained network...')
        checkpoint = torch.load(modelReloadPath, map_location = lambda storage, loc: storage)
        model_dict = model.state_dict()
        checkpoint =  {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
        del checkpoint
        torch.cuda.empty_cache()
        print('Loading Done!\n')
        return model
    else:
        print('\nLoading Fail!\n')
        sys.exit(0)


def modelUpdataSubNet(model, modelOldName, modelOldPath, subNet):
    modelReloadPath = modelOldPath + '/' + modelOldName + '.pkl'
    print('\nOld model path:{}'.format(modelReloadPath))
    if os.path.isfile(modelReloadPath):
        print('Loading previously trained network...')
        # pdb.set_trace()
        checkpoint = torch.load(modelReloadPath, map_location = lambda storage, loc: storage)
        model_dict = model.state_dict()
        model_dict1 = {subNet+'.'+k:v for k, v in model_dict.items()}
        checkpoint =  {k.replace(subNet+'.', ''): v for k, v in checkpoint.items() if k in model_dict1}
        # model_dict = {k.replace(subNet+'.', ''):v for k, v in model_dict.items()}
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
        del checkpoint
        torch.cuda.empty_cache()
        print('Loading Done!\n')
        return model
    else:
        print('\nLoading Fail!\n')
        sys.exit(0)


## Load optimizer
## ----------------------------------------
def optimizerUpdata(optimizer, optimizerOldName, optimizerOldPath):
    optimizerReloadPath = optimizerOldPath + '/' + optimizerOldName + '.pkl'
    print('\nOld Optimizer Path：{}'.format(optimizerReloadPath))
    if os.path.isfile(optimizerReloadPath):
        print('Loading previous optimizer...')
        checkpoint = torch.load(optimizerReloadPath, map_location = lambda storage, loc: storage)
        optimizer.load_state_dict(checkpoint)
        del checkpoint
        torch.cuda.empty_cache()
        print('Loading Done!\n')
        return optimizer
    else:
        print('\nLoading Fail!\n')
        sys.exit(0)

