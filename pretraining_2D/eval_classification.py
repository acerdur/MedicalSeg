# %%
from torchvision import transforms
from numpy import asarray, stack, load
import torch
import random
#import torch.nn as nn
from torch.cuda import is_available
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
#from segmentation_models_pytorch import Unet
from segmentation_models_pytorch import utils
from model import LightningClassifierLSTM, LightningSegmentation
import util

import sys
sys.path.append("..")
from segmentation.data import SegmentationDataset2D, DatasetComposition


   
# %%
ckpt_name = "/home/erdurc/punkreas/segmentation/pretraining_2D/models/model_ckpt_deeplabv3plus_resnet50_classification_lstm_2022-05-10_16-46/best_deeplabv3plus_resnet50_classification_lstm_2022-05-10_16-46.ckpt"
model_info = ckpt_name.split('/')[-1].split('_')
model_architecture = model_info[1]
model_name = model_info[2]
loss_name = 'cross_entropy' if 'classification' in ckpt_name else 'dice'
use_lstm = 'lstm' in ckpt_name

# %%
hparams={}
device = torch.device('cuda' if is_available() else 'cpu')

if use_lstm:
    model = LightningClassifierLSTM.load_from_checkpoint(
        ckpt_name,
        map_location=device,
        hparams=hparams,
        model_architecture=model_architecture,
        model=model_name,
        loss_name=loss_name,
        freeze_layers_in_beginning=False,
        use_imagenet_weights=True,
    )
else:
    model = LightningSegmentation.load_from_checkpoint(
        f"./models/model_ckpt_{experiment_name}/best_{experiment_name}.ckpt",
        map_location=devices,
        hparams=hparams,
        model_architecture=model_architecture,
        model=model_name,
        loss_name=loss_name,
        freeze_layers_in_beginning=False,
        use_imagenet_weights=True, #setting this true is important for model structure, weights will be overriden by ckpt
        classification=True
    )

model.eval()

# %%
# model_dict = load(model_name)
# model.load_state_dict(model_dict)

# %%
train_resolution = 256
std = random.uniform(5.0,15.0) ** 0.5 # imitating the Albumentations GaussNoise
augmentations = {'rotate':[], 'hflip':[], 'vflip':[], 'resize': [train_resolution,train_resolution], 'GaussianNoise': {'mean':0, 'std': std} }


val_indices = load('/home/erdurc/punkreas/segmentation/datasets/MSD/val_idx.npy')
val = SegmentationDataset2D(
    dataroot="/home/erdurc/punkreas/segmentation/datasets/MSD",
    creation_transform=None,
    loading_transform= {'resize': [256,256]},
    indices_3d=val_indices.tolist(),
    mode='classification',
    output_type= 'sequence' if use_lstm else 'single',
    is_train=False,
    return_scan_name=False
)

dataloader = DataLoader(val, batch_size=1, shuffle=False, num_workers=0)
aa = next(iter(dataloader))
import pdb; pdb.set_trace()
# %%
# %%
test_epoch = utils.train.ValidEpoch(
    model=model,
    loss=util.CrossEntropy(use_lstm=use_lstm),
    metrics=[util.Accuracy(use_lstm=use_lstm), util.Precision(use_lstm=use_lstm), util.Recall(use_lstm=use_lstm)],
    device=device,
)
# %%
logs = test_epoch.run(dataloader)
# %%
# i = 0
# for layer in model.modules():
#     if isinstance(layer, nn.Conv2d):
#         i += 1
# print(i)
# # %%
# pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(pytorch_total_params)
# %%
# from plot_stuff import plot_imgs

# # %%
# indcs = []
# for i in range(len(val)):
#     if val[i][1].sum() > 100.0:
#         indcs.append(i)
# print(len(indcs))


# # # %%
# # # %%
# # %%
# imgs, preds, tgts = [], [], []
# for i in indcs[:20]:
#     img, tgt = val[i]
#     pred = model(img.to(device).unsqueeze(0))
#     imgs.append(img.cpu().numpy())
#     tgts.append(tgt.numpy())
#     preds.append(pred.cpu().detach().numpy())
# # %%
# plot_imgs(stack(imgs).squeeze(), stack(tgts).squeeze(), stack(preds).squeeze())

# %%
