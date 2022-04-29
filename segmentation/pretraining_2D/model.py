import torchvision
from torch import nn, optim, stack, normal, reshape, cat, where
from pytorch_lightning import LightningModule
from segmentation_models_pytorch import (
    Unet,
    FPN,
    UnetPlusPlus,
    Linknet,
    PSPNet,
    PAN,
    DeepLabV3,
    DeepLabV3Plus,
)
from segmentation_models_pytorch.utils.losses import DiceLoss, JaccardLoss
from monai.losses import DiceCELoss
from batchnorm_conversion import convert_batchnorm_modules
from itertools import chain
from math import ceil
from copy import deepcopy

import torch.nn.functional as F


class ConvBnElu(nn.Module):
    """
    Conv-Batchnorm-Elu block
    """

    def __init__(self, old_filters, filters, kernel_size=3, strides=1, dilation_rate=1):
        super(ConvBnElu, self).__init__()

        # Conv
        # 'SAME' padding => Output-Dim = Input-Dim/stride -> exact calculation: if uneven add more padding to the right
        # int() floors padding
        # TODO: how to add asymmetric padding? tuple option for padding only specifies the different dims
        same_padding = int(dilation_rate * (kernel_size - 1) * 0.5)

        # TODO: kernel_initializer="he_uniform",

        self.conv = nn.Conv2d(
            in_channels=old_filters,
            out_channels=filters,
            kernel_size=kernel_size,
            stride=strides,
            padding=same_padding,
            dilation=dilation_rate,
            bias=False,
        )

        # BatchNorm
        self.batch_norm = nn.BatchNorm2d(filters)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = F.elu(out)
        return out


class deconv(nn.Module):
    """
    Transposed Conv. with BatchNorm and ELU-activation
    Deconv upsampling of x. Doubles x and y dimension and maintains z.
    """

    def __init__(self, old_filters):
        super(deconv, self).__init__()

        kernel_size = 4
        stride = 2
        dilation_rate = 1

        # TODO: how to add asymmetric padding? possibly use "output_padding here"
        same_padding = int(dilation_rate * (kernel_size - 1) * 0.5)

        # TODO: kernel_initializer="he_uniform",

        self.transp_conv = nn.ConvTranspose2d(
            in_channels=old_filters,
            out_channels=old_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=same_padding,
            bias=False,
        )

        self.batch_norm = nn.BatchNorm2d(old_filters)

    def forward(self, x):
        out = self.transp_conv(x)
        out = self.batch_norm(out)
        out = F.elu(out)
        return out


class repeat_block(nn.Module):
    """
    RDDC - Block
    Reccurent conv block with decreasing kernel size.
    Makes use of atrous convolutions to make large kernel sizes computationally feasible

    """

    def __init__(self, in_filters, out_filters, dropout=0.2):
        super(repeat_block, self).__init__()

        # Skip connection
        # TODO: Reformatting necessary?

        self.convBnElu1 = ConvBnElu(in_filters, out_filters, dilation_rate=4)
        self.dropout1 = nn.Dropout2d(dropout)
        self.convBnElu2 = ConvBnElu(out_filters, out_filters, dilation_rate=3)
        self.dropout2 = nn.Dropout2d(dropout)
        self.convBnElu3 = ConvBnElu(out_filters, out_filters, dilation_rate=2)
        self.dropout3 = nn.Dropout2d(dropout)
        self.convBnElu4 = ConvBnElu(out_filters, out_filters, dilation_rate=1)

    def forward(self, x):
        skip1 = x
        out = self.convBnElu1(x)
        out = self.dropout1(out)
        out = self.convBnElu2(out + skip1)
        out = self.dropout2(out)
        skip2 = out
        out = self.convBnElu3(out)
        out = self.dropout3(out)
        out = self.convBnElu4(out + skip2)

        # TODO: In this implementation there was again a skip connection from first input, not shown in paper however?
        out = skip1 + out
        return out


class MoNet(nn.Module):
    def __init__(
        self,
        input_shape=(1, 256, 256),
        output_classes=1,
        depth=2,
        n_filters_init=16,
        dropout_enc=0.2,
        dropout_dec=0.2,
        activation=None,
    ):
        super(MoNet, self).__init__()

        # store param in case they're needed later
        self.input_shape = input_shape
        self.output_classes = output_classes
        self.depth = depth
        self.features = n_filters_init
        self.dropout_enc = dropout_enc
        self.dropout_dec = dropout_dec

        # encoder
        encoder_list = []

        old_filters = 1
        features = n_filters_init
        for i in range(depth):
            encoder_list.append(
                [f"Enc_ConvBnElu_Before_{i}", ConvBnElu(old_filters, features)]
            )
            old_filters = features
            encoder_list.append(
                [
                    f"Enc_RDDC_{i}",
                    repeat_block(old_filters, features, dropout=dropout_enc),
                ]
            )
            encoder_list.append(
                [
                    f"Enc_ConvBnElu_After_{i}",
                    ConvBnElu(old_filters, features, kernel_size=4, strides=2),
                ]
            )
            features *= 2

        # ModulList instead of Sequential because we don't want the layers to be connected yet
        # we still need to add the skip connections. Dict to know when to add skip connection in forward
        self.encoder = nn.ModuleDict(encoder_list)

        # bottleneck
        bottleneck_list = []
        bottleneck_list.append(ConvBnElu(old_filters, features))
        old_filters = features
        bottleneck_list.append(repeat_block(old_filters, features))

        self.bottleneck = nn.Sequential(*bottleneck_list)

        # decoder
        decoder_list = []
        for i in reversed(range(depth)):
            features //= 2
            decoder_list.append([f"Dec_deconv_Before_{i}", deconv(old_filters)])
            # deconv maintains number of channels
            decoder_list.append(
                [f"Dec_ConvBnElu_{i}", ConvBnElu(old_filters, features)]
            )
            old_filters = features
            decoder_list.append(
                [
                    f"Dec_RDDC_{i}",
                    repeat_block(old_filters, features, dropout=dropout_dec),
                ]
            )

        self.decoder = nn.ModuleDict(decoder_list)

        # head
        head_list = []
        # TODO: kernel_initializer="he_uniform",
        head_list.append(
            nn.Conv2d(
                in_channels=old_filters,
                out_channels=output_classes,
                kernel_size=1,
                stride=1,
                bias=False,
            )
        )

        head_list.append(nn.BatchNorm2d(output_classes))

        # TODO: Consider nn.logsoftmax --> works with NLLoss out of the box --> what we want to use.
        # if output_classes > 1:
        #    activation = nn.Softmax(dim=1)
        # else:
        #    activation = nn.Sigmoid()
        # head_list.append(activation)
        # BCELoss doesn't include sigmoid layer (not as in CELoss)
        # BCELoss can't handle negative number so no log-space
        # activation = nn.Sigmoid()
        # head_list.append(activation)
        # INSTEAD: Added BCEWithLogitsLoss which combines both in a numerically stable way sssss

        self.header = nn.Sequential(*head_list)
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = None

    def forward(self, x):
        skip = []

        # encoder
        out = x
        for key in self.encoder:
            out = self.encoder[key](out)
            if key == "RDDC":
                skip.append(out)

        # bottleneck
        out = self.bottleneck(out)

        # decoder
        for key in self.decoder:
            out = self.decoder[key](out)
            if key == "deconv":
                # Concatenate along channel-dim (last dim)
                # skip.pop() -> get last element and remove it
                out = cat((out, skip.pop()), dim=-1)

        # header
        out = self.header(out)

        if self.activation:
            out = self.activation(out)

        return out


class ResNetGroupNorm(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, 1)
        self.resnet = convert_batchnorm_modules(self.resnet)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.hparams = hparams

    def forward(self, x):
        return self.resnet(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.resnet(data)
        # print(data.dtype)
        # print(target.dtype)
        # print(output.dtype)
        loss = self.loss_fn(output, target.unsqueeze(1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.resnet(data)
        loss = self.loss_fn(output, target.unsqueeze(1))
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def on_epoch_start(self):
        if self.current_epoch == 0:
            for parameter in self.resnet.parameters():
                parameter.requires_grad = False
            bn_layers = []
            for layer_name, layer in self.resnet.named_modules():
                if (
                    isinstance(layer, nn.BatchNorm2d)
                    or isinstance(layer, nn.GroupNorm)
                    or isinstance(layer, nn.Linear)
                ):
                    bn_layers.append(layer_name)
            for bns in bn_layers:
                layers = bns.split(".")
                bn = self.resnet
                semi_last_layer = self.resnet
                for i, l in enumerate(layers):
                    bn = bn.__getattr__(l)
                for p in bn.parameters():
                    p.requires_grad = True
            self.trainer.lr_scheduler = optim.lr_scheduler.StepLR(  ## just keep it constant
                self.trainer.optimizers[0], 1e30, gamma=1
            )
            print("Trained layers: ")
            for name, m in self.resnet.named_parameters():
                if m.requires_grad:
                    print(f" - {name}")
        if self.current_epoch == self.hparams.unfreeze_epochs:
            for parameter in self.resnet.parameters():
                parameter.requires_grad = True
            self.trainer.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.trainer.optimizers[0],
                mode="min",
                patience=self.hparams.lr_reduce_patience,
                factor=self.hparams.lr_reduce_factor,
            )
class IdentityDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Identity()
    
    def forward(self,*inputs):
        return self.layer(inputs[0])

def get_segmentation_block(model, model_architecture, model_args):
    if model in [
        "resnet18",
        "vgg11",
        "vgg11_bn",
        "mobilenet_v2",
        "resnet34",
        "resnet50",
        "resnet101",
    ]:
        if model_architecture == "unet":
            model_class = Unet
        elif model_architecture == "unetpp":
            model_class = UnetPlusPlus
        elif model_architecture == "linknet":
            model_class = Linknet
        elif model_architecture == "fpn":
            model_class = FPN
        elif model_architecture == "psp":
            model_class = PSPNet
        elif model_architecture == "pan":
            model_class = PAN
        elif model_architecture == "deeplabv3":
            model_class = DeepLabV3
        elif model_architecture == "deeplabv3plus":
            model_class = DeepLabV3Plus
        else:
            raise RuntimeError(f"Model {model_architecture} not supported")
        
        return model_class(**model_args)

    #elif "monet" in model:
    #    return MoNet(activation="sigmoid",)
    else:
        raise ValueError(f"Model {model} not supported")

class LightningSegmentation(LightningModule):
    def __init__(
        self,
        hparams,
        model,
        model_architecture,
        loss_name,
        freeze_layers_in_beginning=True,
        use_imagenet_weights=True,
        classification=False,
        **kwargs,
    ):
        super().__init__()
        self.hparams.update(hparams)
        self.freeze_layers_in_beginning = freeze_layers_in_beginning
        self.use_imagenet = use_imagenet_weights
        self.classification = classification

        aux_params = {
            "classes": 2,
            "dropout": 0,
            #"activation": "softmax"
        }

        model_args = {
            "encoder_name": model,
            "in_channels": 3 if self.use_imagenet else 1,
            "classes": 1,
            "activation": "sigmoid",
            "encoder_weights": "imagenet" if self.use_imagenet else None,
            "aux_params": aux_params if self.classification else None
        }

        input_conv = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1, bias=False),
                #nn.InstanceNorm2d(3),
                #nn.ReLU()
            )
            
        self.input_layer = input_conv if self.use_imagenet else nn.Identity()
        self.model = get_segmentation_block(model, model_architecture, model_args)

        #reducing some redundant computation 
        if self.classification:
            self.model.decoder = IdentityDecoder()
            self.model.segmentation_head = nn.Identity()

        if loss_name == "dice":
            self.loss_fn = DiceLoss()
        elif loss_name == "jaccard":
            self.loss_fn = JaccardLoss()
        elif loss_name == "cross_entropy":
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Loss {loss_name} not supported")

    def forward(self, x):
        x = self.input_layer(x)
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        data, target, _= batch
        output = self(data)
        if self.classification:
            output = output[1]
            target = target.flatten()
        else:
            output = reshape(output, target.shape)
        loss = self.loss_fn(output, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target, _ = batch
        output = self(data)
        if self.classification:
            output = output[1]
            target = target.flatten()
        else:
            output = reshape(output, target.shape)
        loss = self.loss_fn(output, target)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        if self.use_imagenet:
            other_params = [param for name,param in self.named_parameters() if 'encoder' not in name]
            param_groups = [
                { 'params': self.model.encoder.parameters(), 'lr': self.hparams.lr/10 },
                { 'params': other_params, 'lr': self.hparams.lr }
            ]
            optimizer = optim.Adam(param_groups)
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def on_epoch_start(self):
        if not self.freeze_layers_in_beginning:
            return
        if self.current_epoch == 0:
            print("Freezing parameters")
            self.frozen_params = []
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad:
                    self.frozen_params.append(name)
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            bn_layers = []
            for layer_name, layer in self.model.named_modules():
                if (
                    isinstance(layer, nn.BatchNorm2d)
                    or isinstance(layer, nn.GroupNorm)
                    or isinstance(layer, nn.Linear)
                ):
                    bn_layers.append(layer_name)
            for bns in bn_layers:
                layers = bns.split(".")
                bn = self.model
                # semi_last_layer = self.model
                for l in layers:
                    bn = bn.__getattr__(l)
                for p in bn.parameters():
                    p.requires_grad = True
            self.trainer.lr_scheduler = optim.lr_scheduler.StepLR(  ## just keep it constant
                self.trainer.optimizers[0], 1e30, gamma=1
            )
            # print("Trained layers: ")
            # for name, m in chain(self.input_layer.named_parameters(), self.model.named_parameters()):
            #     if m.requires_grad:
            #         print(f" - {name}")
        if self.current_epoch == self.hparams.unfreeze_epochs:
            print("Unfreezing parameters")
            for name, parameter in self.model.named_parameters():
                if name in self.frozen_params:
                    parameter.requires_grad = True
            self.trainer.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.trainer.optimizers[0],
                mode="min",
                patience=self.hparams.lr_reduce_patience,
                factor=self.hparams.lr_reduce_factor,
            )

class LightningClassifierLSTM(LightningModule):
    def __init__(
        self,
        hparams,
        model,
        model_architecture,
        loss_name,
        freeze_layers_in_beginning=True,
        use_imagenet_weights=True,
        **kwargs,
    ):
        super().__init__()
        self.hparams.update(hparams)
        self.freeze_layers_in_beginning = freeze_layers_in_beginning
        self.use_imagenet = use_imagenet_weights

        model_args = {
            "encoder_name": model,
            "in_channels": 3 if self.use_imagenet else 1,
            "classes": 1,
            "activation": "sigmoid",
            "encoder_weights": "imagenet" if self.use_imagenet else None,
        }

        dropout = 0.2
        final_activation = {'sigmoid': nn.Sigmoid(), 'softmax': nn.Softmax(dim=1), 'none': nn.Identity()}

        input_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(3),
            nn.ReLU()
        )
        model_block = get_segmentation_block(model, model_architecture, model_args)
        
        self.input_layer = input_conv if self.use_imagenet else nn.Identity()
        self.encoder = model_block.encoder
        model_block = None

        self.pre_lstm = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1)
        )
        
        self.lstm = nn.LSTM(input_size=self.encoder.out_channels[-1], hidden_size=1024, batch_first=True)
        self.classification_head = nn.Sequential(
            nn.Linear(in_features=self.lstm.hidden_size, out_features=self.lstm.hidden_size),
            nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity(),
            nn.Linear(in_features=self.lstm.hidden_size ,out_features=2),
            final_activation['none']
        )
        
        if loss_name == "cross_entropy":
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Loss {loss_name} not supported")


    def forward(self, x):
        """
        x: torch.Tensor     batch_size x seq_len x 1 x H x W    Input 3D scan by slices
        out: torch.Tensor   batch_size x seq_len x 2            Class logits for 0,1 (pancreas in / not)
        """
        B,T,C,H,W = x.shape
        seq = []
        for t in range(T):
            features = self.input_layer(x[:,t,:,:,:])
            features = self.encoder(features)
            slc = self.pre_lstm(features[-1])
            seq.append(slc.unsqueeze(1))
        seq = cat(seq, dim=1)
        ## DEPRECATED & changed to a loop because of high memory usage during inference 
        # x = self.input_layer(x.view(-1,C,H,W))
        # features = self.encoder(x)
        # seq = self.pre_lstm(features[-1]).view(B,T,-1)
        out, states = self.lstm(seq)
        out = self.classification_head(out)
        return states, out

    def soft_pred(self,x,window_size_mult_of=None):
        """
        Fill in the window between first & last 1 predictions with 1s
        Works only with batch_size = 1

        window_size_mult_of: select from [2,4,8,16] so that the output window of 1s has a length
                              that is multiple of given number
        """
        lstm_states, preds = self(x)
        lstm_states = None
        preds = preds.argmax(dim=2).flatten()
        zmin, zmax = where(preds)[0][[0,-1]] #[i for i,x in enumerate(preds[0]) if x != 0 ]
        zmin, zmax = zmin.item(), zmax.item()
        
        max_size = preds.shape[0]
        #import pdb; pdb.set_trace()
        window_size = preds.sum().item()
        gap = 0
        if window_size_mult_of:
            target_size = window_size_mult_of * ceil(window_size / window_size_mult_of)
            gap = target_size - window_size
        ### mark indices to start & end the window of 1s 
        start_idx = zmin - gap // 2
        end_idx = zmax + ((gap // 2) + (gap % 2)) + 1
        if end_idx > max_size:
            start_idx -= (end_idx - max_size)
        if start_idx < 0:
            end_idx += (-start_idx)  # if the start index exceeds slice limits (below 0), add rest to end_index
         # if end index exceeds limits, add rest of the slices to start
        ### most probably the two conditions won't be True at the same time
        ### but be careful with small scans & large window_size_mult_of (e.g. 16)
        preds[max(0,start_idx):min(end_idx,max_size)] = 1

        import pdb; pdb.set_trace()
        return preds, (start_idx, end_idx)

    def training_step(self, batch, batch_idx):
        data, target, _ = batch
        lstm_states, output = self(data)
        loss = self.loss_fn(output.view(-1,output.shape[-1]), target.flatten())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target, _ = batch
        lstm_states, output = self(data)
        loss = self.loss_fn(output.view(-1,output.shape[-1]), target.flatten())
        self.log("val_loss", loss)
        return loss
    
    def configure_optimizers(self):
        if self.use_imagenet:
            other_params = [param for name,param in self.named_parameters() if 'encoder' not in name]
            param_groups = [
                { 'params': self.encoder.parameters(), 'lr': self.hparams.lr/10 },
                { 'params': other_params, 'lr': self.hparams.lr }
            ]
            optimizer = optim.Adam(param_groups)
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def on_epoch_start(self):
        if not self.freeze_layers_in_beginning:
            return
        if self.current_epoch == 0:
            print("Freezing parameters")
            self.frozen_params = []
            for name, parameter in self.encoder.named_parameters():
                if parameter.requires_grad:
                    self.frozen_params.append(name)
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            bn_layers = []
            for layer_name, layer in self.encoder.named_modules():
                if (
                    isinstance(layer, nn.BatchNorm2d)
                    or isinstance(layer, nn.GroupNorm)
                    or isinstance(layer, nn.Linear)
                ):
                    bn_layers.append(layer_name)
            for bns in bn_layers:
                layers = bns.split(".")
                bn = self.encoder
                # semi_last_layer = self.model
                for l in layers:
                    bn = bn.__getattr__(l)
                for p in bn.parameters():
                    p.requires_grad = True
            self.trainer.lr_scheduler = optim.lr_scheduler.StepLR(  ## just keep it constant
                self.trainer.optimizers[0], 1e30, gamma=1
            )
            # print("Trained layers: ")
            # for name, m in self.named_parameters():
            #     if m.requires_grad:
            #         print(f" - {name}")
        if self.current_epoch == self.hparams.unfreeze_epochs:
            print("Unfreezing parameters")
            for name, parameter in self.encoder.named_parameters():
                if name in self.frozen_params:
                    parameter.requires_grad = True
            self.trainer.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.trainer.optimizers[0],
                mode="min",
                patience=self.hparams.lr_reduce_patience,
                factor=self.hparams.lr_reduce_factor,
            )    

## 2D LSTM models from https://github.com/HowieMa/lstm_multi_modal_UNet/

class LSTM0(nn.Module):
    def __init__(self, in_c=5, ngf=32):
        super(LSTM0, self).__init__()
        self.conv_gx_lstm0 = nn.Conv2d(in_c + ngf, ngf, kernel_size=3, padding=1)
        self.conv_ix_lstm0 = nn.Conv2d(in_c + ngf, ngf, kernel_size=3, padding=1)
        self.conv_ox_lstm0 = nn.Conv2d(in_c + ngf, ngf, kernel_size=3, padding=1)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, xt):
        """
        :param xt:      bz * 5(num_class) * 240 * 240
        :return:
            hide_1:    bz * ngf(32) * 240 * 240
            cell_1:    bz * ngf(32) * 240 * 240
        """
        gx = self.conv_gx_lstm0(xt)
        ix = self.conv_ix_lstm0(xt)
        ox = self.conv_ox_lstm0(xt)

        gx = self.tanh(gx)
        ix = self.sigmoid(ix)
        ox = self.sigmoid(ox)

        cell_1 = self.tanh(gx * ix)
        hide_1 = ox * cell_1
        return cell_1, hide_1


class LSTM(nn.Module):
    def __init__(self, in_c=5, ngf=32):
        super(LSTM, self).__init__()
        self.conv_ix_lstm = nn.Conv2d(in_c + ngf, ngf, kernel_size=3, padding=1, bias=True)
        self.conv_ih_lstm = nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False)

        self.conv_fx_lstm = nn.Conv2d(in_c + ngf, ngf, kernel_size=3, padding=1, bias=True)
        self.conv_fh_lstm = nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False)

        self.conv_ox_lstm = nn.Conv2d(in_c + ngf, ngf, kernel_size=3, padding=1, bias=True)
        self.conv_oh_lstm = nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False)

        self.conv_gx_lstm = nn.Conv2d(in_c + ngf, ngf, kernel_size=3, padding=1, bias=True)
        self.conv_gh_lstm = nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, bias=False)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, xt, cell_t_1, hide_t_1):
        """
        :param xt:          bz * (5+32) * 240 * 240
        :param hide_t_1:    bz * ngf(32) * 240 * 240
        :param cell_t_1:    bz * ngf(32) * 240 * 240
        :return:
        """
        gx = self.conv_gx_lstm(xt)         # output: bz * ngf(32) * 240 * 240
        gh = self.conv_gh_lstm(hide_t_1)   # output: bz * ngf(32) * 240 * 240
        g_sum = gx + gh
        gt = self.tanh(g_sum)

        ox = self.conv_ox_lstm(xt)          # output: bz * ngf(32) * 240 * 240
        oh = self.conv_oh_lstm(hide_t_1)    # output: bz * ngf(32) * 240 * 240
        o_sum = ox + oh
        ot = self.sigmoid(o_sum)

        ix = self.conv_ix_lstm(xt)              # output: bz * ngf(32) * 240 * 240
        ih = self.conv_ih_lstm(hide_t_1)        # output: bz * ngf(32) * 240 * 240
        i_sum = ix + ih
        it = self.sigmoid(i_sum)

        fx = self.conv_fx_lstm(xt)              # output: bz * ngf(32) * 240 * 240
        fh = self.conv_fh_lstm(hide_t_1)        # output: bz * ngf(32) * 240 * 240
        f_sum = fx + fh
        ft = self.sigmoid(f_sum)

        cell_t = ft * cell_t_1 + it * gt        # bz * ngf(32) * 240 * 240
        hide_t = ot * self.tanh(cell_t)            # bz * ngf(32) * 240 * 240

        return cell_t, hide_t


class LightningSegmentationLSTM(LightningModule):
    def __init__(
        self,
        hparams,
        model,
        model_architecture,
        loss_name,
        freeze_layers_in_beginning=True,
        use_imagenet_weights=True,
        **kwargs,
    ):
        super().__init__()
        self.hparams.update(hparams)
        self.freeze_layers_in_beginning = freeze_layers_in_beginning
        self.use_imagenet = use_imagenet_weights
        
        model_args = {
            "encoder_name": model,
            "in_channels": 3 if self.use_imagenet else 1,
            "classes": 3,
            "activation": None,
            "encoder_weights": "imagenet" if self.use_imagenet else None,
        }
            
        input_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(3),
            nn.ReLU()
        )
        model_block = get_segmentation_block(model, model_architecture, model_args)
        self.input_layer = input_conv if self.use_imagenet else nn.Identity()
        self.encoder = model_block.encoder
        self.decoder = model_block.decoder
        self.model_head = model_block.segmentation_head
        model_block = None

        self.lstm0 = LSTM0(in_c=0, ngf=self.decoder.out_channels)
        self.lstm = LSTM(in_c=0, ngf=self.decoder.out_channels)
        self.out_head = deepcopy(self.model_head)

        if loss_name == "dice_cross_entropy":
            self.loss_fn = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        else:
            raise ValueError(f"Loss {loss_name} not supported")

    def forward(self, x):
        """
        x:           torch.Tensor    batch_size x seq_len x 1 x H x W           Input 3D scan by slices
        output_2d:   torch.Tensor    batch_size x seq_len x num_class x H x W   Class logits for segmentation, 2D model output
        output_lstm: torch.Tensor    batch_size x seq_len x num_class x H x W   Class logits for segmentation, full model output
        """
        B,T,C,H,W = x.shape
        
        output_lstm = []
        output_2d = []
        cell = None
        hidden = None
        for t in range(T):
            img = x[:,t,:,:,:]  #bs x 1 x H x W
            decoder_out = self.decoder(*self.encoder(self.input_layer(img)))
            out2d = self.model_head(decoder_out)
            output_2d.append(out2d.unsqueeze(1))
            #does not work because out and decoder_out have different spatial size
            #use decoder_out in the lstm for now
            # lstm_in = torch.cat([out2d, decoder_out], dim=1) 

            if t == 0:
                cell, hidden = self.lstm0(decoder_out)
            else:
                cell, hidden = self.lstm(decoder_out, cell, hidden)

            out_lstm = self.out_head(hidden)
            output_lstm.append(out_lstm.unsqueeze(1))

        return cat(output_2d, dim=1), cat(output_lstm, dim=1)

    def inference(self, x, target):
        output2d, output = self(x)
        # adjust dims for MonAI dice metric
        target = target.permute(0,2,3,1).unsqueeze(1)
        output = output.permute(0,2,3,4,1)

        return output, target

    def training_step(self, batch, batch_idx):
        data, target, _ = batch
        output2d, output = self(data)
        #outputs to BxCxHxWxD
        loss = self.loss_fn(output2d.permute(0,2,3,4,1), target.permute(0,2,3,1).unsqueeze(1)) + self.loss_fn(output.permute(0,2,3,4,1), target.permute(0,2,3,1).unsqueeze(1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target, _ = batch
        lstm_states, output = self(data)
        loss = self.loss_fn(output.permute(0,2,3,4,1), target.permute(0,2,3,1).unsqueeze(1))
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        if self.use_imagenet:
            other_params = [param for name,param in self.named_parameters() if 'encoder' not in name]
            param_groups = [
                { 'params': self.encoder.parameters(), 'lr': self.hparams.lr/10 },
                { 'params': other_params, 'lr': self.hparams.lr }
            ]
            optimizer = optim.Adam(param_groups)
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def on_epoch_start(self):
        if not self.freeze_layers_in_beginning:
            return
        if self.current_epoch == 0:
            print("Freezing parameters")
            self.frozen_params = []
            for name, parameter in self.encoder.named_parameters():
                if parameter.requires_grad:
                    self.frozen_params.append(name)
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            bn_layers = []
            for layer_name, layer in self.encoder.named_modules():
                if (
                    isinstance(layer, nn.BatchNorm2d)
                    or isinstance(layer, nn.GroupNorm)
                    or isinstance(layer, nn.Linear)
                ):
                    bn_layers.append(layer_name)
            for bns in bn_layers:
                layers = bns.split(".")
                bn = self.encoder
                # semi_last_layer = self.model
                for l in layers:
                    bn = bn.__getattr__(l)
                for p in bn.parameters():
                    p.requires_grad = True
            self.trainer.lr_scheduler = optim.lr_scheduler.StepLR(  ## just keep it constant
                self.trainer.optimizers[0], 1e30, gamma=1
            )
            # print("Trained layers: ")
            # for name, m in self.named_parameters():
            #     if m.requires_grad:
            #         print(f" - {name}")
        if self.current_epoch == self.hparams.unfreeze_epochs:
            print("Unfreezing parameters")
            for name, parameter in self.encoder.named_parameters():
                if name in self.frozen_params:
                    parameter.requires_grad = True
            self.trainer.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.trainer.optimizers[0],
                mode="min",
                patience=self.hparams.lr_reduce_patience,
                factor=self.hparams.lr_reduce_factor,
            )  

if __name__ == '__main__':
    import torch
    hparams = {
    "lr": 1e-3,
    "unfreeze_epochs": 3,
    "lr_reduce_patience": 5,
    "lr_reduce_factor": 0.5,
    }
    model_name = "resnet50"
    model_architecture = "deeplabv3plus"
    loss_name = "cross_entropy"
    model = LightningSegmentation(
        hparams=hparams,
        model=model_name,
        model_architecture=model_architecture,
        loss_name=loss_name,
        freeze_layers_in_beginning=not "monet" in model_name,
        classification=True
    )
    model2 = LightningClassifierLSTM(
        hparams=hparams,
        model=model_name,
        model_architecture=model_architecture,
        loss_name=loss_name,
        freeze_layers_in_beginning=not "monet" in model_name,
    )
    model3 = LightningSegmentationLSTM(
        hparams=hparams,
        model=model_name,
        model_architecture=model_architecture,
        loss_name=loss_name,
        freeze_layers_in_beginning=not "monet" in model_name,
    )

    dummy_input = torch.rand((8,1,256,256))
    dummy_seq = torch.rand((8,4,1,512,512)) # B x t x 1 x H x W
    
    dummy_target = torch.randint(0,2,(8,1,256,256))
    dummy_lbl = torch.randint(0,2,(8,))
    dummy_lbl_seq = torch.randint(0,2,(8,4,1))

    output = model(dummy_input)
    
    import pdb; pdb.set_trace()