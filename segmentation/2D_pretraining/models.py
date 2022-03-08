import torchvision
from torch import nn, optim, zeros_like, ones_like, stack, normal, reshape
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
from batchnorm_conversion import convert_batchnorm_modules
from itertools import chain

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
                out = torch.cat((out, skip.pop()), dim=-1)

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
    ):
        super().__init__()
        self.hparams.update(hparams)
        self.freeze_layers_in_beginning = freeze_layers_in_beginning
        self.use_imagenet = use_imagenet_weights
        self.classification = classification
        if model in [
            "resnet18",
            "vgg11",
            "vgg11_bn",
            "mobilenet_v2",
            "resnet34",
            "resnet50",
            "resnet101",
        ]:
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
            input_conv = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1, bias=False),
                #nn.InstanceNorm2d(3),
                #nn.ReLU()
            )
            model_block = model_class(**model_args)
            
            self.input_layer = input_conv if self.use_imagenet else nn.Identity()
            self.model = model_block

            #reducing some redundant computation 
            if self.classification:
                self.model.decoder = IdentityDecoder()
                self.model.segmentation_head = nn.Identity()

        elif "monet" in model:
            self.model = MoNet(activation="sigmoid",)
        else:
            raise ValueError(f"Model {model} not supported")
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
        data, target = batch
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
        data, target = batch
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
    ):
        super().__init__()
        self.hparams.update(hparams)
        self.freeze_layers_in_beginning = freeze_layers_in_beginning
        self.use_imagenet = use_imagenet_weights
        if model in [
            "resnet18",
            "vgg11",
            "vgg11_bn",
            "mobilenet_v2",
            "resnet34",
            "resnet50",
            "resnet101",
        ]:

            model_args = {
                "encoder_name": model,
                "in_channels": 3 if self.use_imagenet else 1,
                "classes": 1,
                "activation": "sigmoid",
                "encoder_weights": "imagenet" if self.use_imagenet else None,
            }

            dropout = 0.2
            final_activation = {'sigmoid': nn.Sigmoid(), 'softmax': nn.Softmax(dim=1), 'none': nn.Identity()}

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
            
            input_conv = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1, bias=False),
                nn.InstanceNorm2d(3),
                nn.ReLU()
            )
            model_block = model_class(**model_args)
            
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
           
        else:
            raise ValueError(f"Model {model} not supported")

        if loss_name == "cross_entropy":
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Loss {loss_name} not supported")


    def forward(self, x):
        B,T,C,H,W = x.shape
        x = self.input_layer(x.view(-1,C,H,W))
        features = self.encoder(x)
        seq = self.pre_lstm(features[-1]).view(B,T,-1)
        out, states = self.lstm(seq)
        out = self.classification_head(out)
        return states, out

    def training_step(self, batch, batch_idx):
        data, target = batch
        lstm_states, output = self(data)
        loss = self.loss_fn(output.view(-1,output.shape[-1]), target.flatten())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
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
    dummy_input = torch.rand((8,1,256,256))
    dummy_seq = torch.rand((8,4,1,512,512)) # B x t x 1 x H x W
    
    dummy_target = torch.randint(0,2,(8,1,256,256))
    dummy_lbl = torch.randint(0,2,(8,))
    dummy_lbl_seq = torch.randint(0,2,(8,4,1))

    output = model(dummy_input)

    import segmentation_models_pytorch.encoders 
    encoder =  model.model.encoder
    import pdb; pdb.set_trace()