# %%
from torchvision import transforms
from numpy import asarray, stack, load
from torch import nn, no_grad, sigmoid, device
from torch.cuda import is_available
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
#from segmentation_models_pytorch import Unet
from segmentation_models_pytorch import utils
from model import LightningSegmentation

import sys
sys.path.append("..")
from data import SegmentationDataset2D, DatasetComposition

# %%
ckpt_name = "/home/erdurc/punkreas/segmentation/2D_pretraining/models/model_ckpt_deeplabv3plus_resnet50_dice_2022-02-28_15-30/best_deeplabv3plus_resnet50_dice_2022-02-28_15-30.ckpt"
model_info = ckpt_name.split('/')[-1].split('_')
model_architecture = model_info[1]
model_name = model_info[2]
loss_name = model_info[3]

# %%
hparams={}
devices = device('cuda' if is_available() else 'cpu')

model = LightningSegmentation.load_from_checkpoint(
    ckpt_name,
    map_location=devices,
    hparams=hparams,
    model_architecture=model_architecture,
    model=model_name,
    loss_name=loss_name,
    freeze_layers_in_beginning=False,
    use_imagenet_weights=True,
)
model.eval()

# %%
# model_dict = load(model_name)
# model.load_state_dict(model_dict)

# %%
train_resolution = 512

val_indices = load('/home/erdurc/punkreas/segmentation/checkpoints/medical_resnet50_pretrain/val_idx.npy')
val = SegmentationDataset2D(
    dataroot="/home/erdurc/punkreas/segmentation/datasets/MSD",
    creation_transform=None,
    indices_3d=val_indices.tolist(),
    mode='segmentation',
    return_scan_name = False
)

dataloader = DataLoader(val, batch_size=1, shuffle=False)
import pdb; pdb.set_trace()
# %%
# %%
test_epoch = utils.train.ValidEpoch(
    model=model,
    loss=utils.losses.DiceLoss(eps=1e-7, beta=1.0),
    metrics=[utils.metrics.Fscore(eps=1e-7), utils.metrics.IoU()],
    device=devices,
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
