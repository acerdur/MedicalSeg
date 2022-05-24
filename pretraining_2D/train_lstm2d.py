# %%
from deepee.surgery import SurgicalProcedures
import pytorch_lightning as pl
import torchvision
import sys
import random
import numpy as np
from numpy import float32 as npfloat32
from torch import nn, float32 as torchfloat32, optim, save
from torch.cuda import is_available
from torch.utils.data import random_split, DataLoader, sampler
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from model import LightningClassifierLSTM, LightningSegmentationLSTM
from datetime import datetime
from deepee import ModelSurgeon
from pytorch_lightning.utilities.seed import seed_everything

from punkreas import transform
sys.path.append("..")
from data import SegmentationDataset2D, DatasetComposition

# %%
# settings
seed_everything(0,workers=True)

dataroot = '/home/erdurc/punkreas/segmentation/datasets/MSD'
train_resolution = 256
batch_size = 4
model_name = "resnet50"
model_architecture = "deeplabv3plus"
loss_name = "cross_entropy"
convert_to_group_norm = False
task = 'classification'
assert model_name in [
    "resnet18",
    "resnet34",
    "resnet101",
    "resnet50",
    "monet",
    "monet_tanh",
    "monet_bnnostats",
    "monet_gn",
    "monet_in",
    "monet_online",
    "vgg11",
    "vgg11_bn",
    "mobilenet_v2",
]
assert loss_name in ["dice", "jaccard", "cross_entropy", "dice_cross_entropy"]
if loss_name == 'cross_entropy':
    assert task == 'classification'
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
experiment_name = f"{model_architecture}_{model_name}_{task}_lstm_{timestamp}"
# %%
# actually unnecessary transformations, as they were done before saving the slices as PNGs
# so these won't be used now. useful if the dataset has to be recreated 
creation_transforms = {
    'Clip': {'amin': -150, 'amax': 250},
    'Normalize': {'bounds': [-150, 250]},
    #'Resize': {'output_shape': [train_resolution, train_resolution, 'z']},
    'ToReferencePosition': {}
                  }

creation_transformations = [
        transform.Compose([getattr(transform,name)(**options) for name,options in creation_transforms.items()])
        ]

std = random.uniform(5.0,15.0) ** 0.5 # imitating the Albumentations GaussNoise
augmentations = {'rotate':[], 'hflip':[], 'vflip':[], 'resize': [train_resolution,train_resolution], } #'GaussianNoise': {'mean':0, 'std': std} }

# %%
train_indices = np.load('/home/erdurc/punkreas/segmentation/datasets/MSD/train_idx.npy')
val_indices = np.load('/home/erdurc/punkreas/segmentation/datasets/MSD/val_idx.npy')
train = SegmentationDataset2D(
    dataroot="/home/erdurc/punkreas/segmentation/datasets/MSD",
    creation_transform=creation_transformations[0],
    loading_transform=augmentations,
    indices_3d=train_indices.tolist(),
    mode=task,
    output_type='sequence',
    temporal=4,
)
val = SegmentationDataset2D(
    dataroot="/home/erdurc/punkreas/segmentation/datasets/MSD",
    creation_transform=creation_transformations[0],
    loading_transform={'resize': [train_resolution,train_resolution]},
    indices_3d=val_indices.tolist(),
    mode=task,
    output_type='sequence',
    is_train=False,
)
print(f" - {len(train)} training images\n - {len(val)} validation images")
# %%
train, val = (
    DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True
    ),
    DataLoader(
        val, batch_size=1, shuffle=False, num_workers=16, pin_memory=True
    ),
)

# %%
logger = pl.loggers.TensorBoardLogger("tb_logs", name=f"{experiment_name}")
# %%
trainer = pl.Trainer(
    gpus=1,
    logger=logger,
    auto_lr_find=True,
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            dirpath=f"./models/model_ckpt_{experiment_name}/",
            filename=f"best_{experiment_name}",
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min",
        ),
        pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", patience=20),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ],
    track_grad_norm=2,
    progress_bar_refresh_rate=0,
    val_check_interval=0.5,
    # overfit_batches=1,
    # max_epochs=1,
)
# %%
# model = LightningSegmentation(
#     hparams={
#         "lr": 1,
#         "unfreeze_epochs": 1,
#         "lr_reduce_patience": 5,
#         "lr_reduce_factor": 0.5,
#     },
#     model=model_name,
#     freeze_layers_in_beginning=False,
# )
# lr_finder = trainer.tuner.lr_find(model, train, val)
# # print(lr_finder.results)
# fig = lr_finder.plot(suggest=True)
# fig.show()
# new_lr = lr_finder.suggestion()
# print(new_lr)

# %%
hparams = {
    "lr": 1e-3,
    "unfreeze_epochs": 3,
    "lr_reduce_patience": 5,
    "lr_reduce_factor": 0.5,
    "weight_decay": 1e-4
}
if task == 'classification':
    model = LightningClassifierLSTM(
        hparams,
        model_architecture=model_architecture,
        model=model_name,
        loss_name=loss_name,
        freeze_layers_in_beginning=not "monet" in model_name,
        use_imagenet_weights=True
    )
elif task == 'segmentation':
    model = LightningSegmentationLSTM(
        hparams,
        model_architecture=model_architecture,
        model=model_name,
        loss_name=loss_name,
        freeze_layers_in_beginning=not "monet" in model_name,
        use_imagenet_weights=True
    )
# %%
if convert_to_group_norm:
    surgeon = ModelSurgeon(SurgicalProcedures.BN_to_GN)
    surgeon.operate(model.model)


# %%
#print(model.hparams)

# %%
trainer.fit(model, train, val)
# %%
model = LightningClassifierLSTM.load_from_checkpoint(
    f"./models/model_ckpt_{experiment_name}/best_{experiment_name}.ckpt",
    hparams=hparams,
    model_architecture=model_architecture,
    model=model_name,
    loss_name=loss_name,
    freeze_layers_in_beginning=False,
    use_imagenet_weights=True, #setting this true is important for model structure, weights will be overriden by ckpt
)
# %%
save(model.state_dict(), f"models/best_{experiment_name}.pt")

# %%
