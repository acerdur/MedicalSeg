"""
Important setup functions that are used in the training script
"""
import os
import yaml
import torch
import numpy as np
#from skimage.transform import resize
from scipy import ndimage
from torch.nn import init
from torch.utils.data import DataLoader, random_split, sampler
from typing import Mapping, Optional, Union
from torch.optim import lr_scheduler

import models
from punkreas import transform, optimizer
from data import SegmentationDataset, DatasetComposition




def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def get_model(opt):
    """
    Define model, move to device & load pretrained layers.
    Parameters:
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
    """
    # obtain the model
    if opt.baseline == 'medicalnet':
        model = getattr(models,f'ResNet{opt.model_depth}')(opt) # create ResNet from options
    elif opt.baseline == 'monainet':
        model = getattr(models,f'{opt.model}')(opt) # create UNet or UNetR from options 
        # for now Monainet options are not specified, rather fixed in module

    # initialize model weights
    init_weights(model, init_type=opt.init_type, init_gain=opt.init_gain)

    # get device name: CPU or GPU
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')  
    # move model to device
    if len(opt.gpu_ids) > 0:
        assert (torch.cuda.is_available())
        model = model.to(device)
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)  
    else:
        model = model.to(device)
            
    # load pretrained
    if opt.pretrain_path:
        pretrain = torch.load(opt.pretrain_path, map_location=device)
        model.load_state_dict(pretrain, strict=False)

        if opt.memory_format == 'channels_last':
            model = model.to(memory_format=torch.channels_last)

        # if using pretrained model and new layers are specified
        # add them to new_parameters so that they have higher learning rate
        new_parameters = []
        for pname, p in model.named_parameters():
            for layer_name in opt.new_layer_names:
                if pname.find(layer_name) >= 0:
                    new_parameters.append(p)
                    break
    else:
        # without pretrained layers, treat every parameter as new 
        new_parameters = []
        for pname, p in model.named_parameters():
            new_parameters.append(p)
    
    new_parameters_id = list(map(id, new_parameters))
    base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
    parameters = {'base_parameters': base_parameters, 
                    'new_parameters': new_parameters}    

    return model, parameters

def get_scheduler(opt, optimizer):
    """
    Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def get_optimizer(opt, parameters):
    """
    Return the optimizer 

    Parameters:
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        parameters         -- dict of Pytorch model parameters, returned by get_model function

    """
    params = [
            { 'params': parameters['base_parameters'], 'lr': opt.lr/100, 'weight_decay': opt.weight_decay }, 
            { 'params': parameters['new_parameters'], 'lr': opt.lr, 'weight_decay': opt.weight_decay }
            ]
    # add the momentum value to the parameter dictionaries
    if opt.optimizer == 'SGD':
        for param_group in params:
            param_group['momentum'] = opt.betas[0] 
    else:
        for param_group in params:
            param_group['betas'] = tuple(opt.betas)

    optim = getattr(optimizer, opt.optimizer)(params)

    return optim

def get_dataloaders(opt):

    with open(os.path.abspath(opt.data_config)) as yaml_file:
        dataconfig = yaml.full_load(yaml_file)

    dataconfig = dataconfig['data'] # dump other categories if given and reduce one level of nest
    dataset_paths = [ os.path.abspath(pth) for pth in dataconfig['datasets']]

    # Configure transformations
    transformations = [
        transform.Compose([getattr(transform,name)(**options) for name,options in dataconfig['transform'].items()])
        for _ in range(len(dataset_paths))
        ]
    # Create dataset composition from all datasets
    dataset_composition = DatasetComposition([
        SegmentationDataset.from_compressed_folder(dataset_folder, transformations[i]) 
        for i, dataset_folder in enumerate(dataset_paths)
    ])


    # Split data into training, validation 
    # save split indices for later use (reproducibility)
    indices = list(range(len(dataset_composition)))
    val_split = dataconfig["validation"]["share"]
    np.random.shuffle(indices)
    train_indices, val_indices = indices[val_split:], indices[:val_split]
    np.save(os.path.join(opt.checkpoints_dir,opt.name,'train_idx.npy'),train_indices)
    np.save(os.path.join(opt.checkpoints_dir,opt.name,'val_idx.npy'),val_indices)
    train_sampler = sampler.SubsetRandomSampler(train_indices)
    val_sampler = sampler.SubsetRandomSampler(val_indices)
   
    # train_data, val_data = random_split(
    #     dataset_composition,
    #     [
    #         dataconfig["training"]["share"],
    #         dataconfig["validation"]["share"],
    #     ],
    # )

    # Configure dataloaders
    train_loader: Optional[DataLoader] = DataLoader(
        dataset_composition,
        batch_size=dataconfig["training"]["batch_size"],
        num_workers=opt.num_workers,
        sampler=train_sampler
    )
    if dataconfig["validation"]["share"]:
        val_loader: Optional[DataLoader] = DataLoader(
            dataset_composition,
            batch_size=dataconfig["validation"]["batch_size"],
            num_workers=opt.num_workers,
            sampler=val_sampler
        )
    else:
        val_loader = None

    return train_loader, val_loader
    

def prepare_for_loss(opt,outputs,targets):
    
    if opt.baseline == 'monainet':
        if opt.loss_function == 'cross_entropy':
            outputs = outputs.permute(0,1,4,2,3)
            new_targets = targets.permute(0,3,1,2)
        else:
            new_targets = targets.unsqueeze(1)

    elif opt.baseline == 'medicalnet':
        [n, _, d, h, w] = outputs.shape
        
        if opt.loss_function == 'cross_entropy':
            new_targets = np.zeros([n,d,h,w])
            for batch_id in range(n):
                target = targets[batch_id] 
                [ori_d, ori_h, ori_w] = target.shape
                scale = [d*1.0/ori_d, h*1.0/ori_h, w*1.0/ori_w]
                target = ndimage.zoom(target.cpu(), scale, order=0)
                new_targets[batch_id] = target
        else:
            new_targets = np.zeros([n,h,w,d])
            for batch_id in range(n):
                target = targets[batch_id]
                [ori_h, ori_w, ori_d] = target.shape
                scale = [h*1.0/ori_h, w*1.0/ori_w, d*1.0/ori_d]
                target = ndimage.zoom(target.cpu(), scale, order=0)
                new_targets[batch_id] = target
            new_targets = np.expand_dims(new_targets,axis=1)
            outputs = outputs.permute(0,1,3,4,2)

        new_targets = torch.Tensor(new_targets).to(torch.long)
        if not opt.no_cuda:
            new_targets = new_targets.cuda()

    return outputs, new_targets