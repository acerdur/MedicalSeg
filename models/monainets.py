from monai.networks.nets import UNet, SegResNet, VNet, DynUNet
from torch import nn

class UNet(UNet):
    def __init__(self, opt):
        self.opt = opt
        super().__init__(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2
        )
    def inference(self,x,target):
        ### output for MonAI DiceMetric
        out = self(x)
        target = target.unsqueeze(0)

        return out, target

class SegResNet(SegResNet):
    def __init__(self, opt):
        self.opt = opt
        super().__init__(
            spatial_dims=3,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=8,
            in_channels=1,
            out_channels=3,
            dropout_prob=0.2,
        )
    def inference(self,x,target):
        out = self(x)
        target = target.unsqueeze(0)

        return out, target

class VNet(VNet):
    def __init__(self, opt):
        self.opt = opt
        super().__init__(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            dropout_prob=0.2,
        )
    def inference(self,x,target):
        out = self(x)
        target = target.unsqueeze(0)

        return out, target

class DynUNet(DynUNet):
    def __init__(self, opt):
        self.opt = opt
        kernels = [[3, 3, 1], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        strides = [[1, 1, 1], [2, 2, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 1]]
        super().__init__(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            norm_name="instance",
            deep_supervision=True,
            deep_supr_num=3,
        )
    def inference(self,x,target):
        ### output for MonAI DiceMetric
        out = self(x)
        target = target.unsqueeze(0)

        return out, target

def get_kernels_strides(task_id):
    sizes, spacings = [224, 224, 40], [0.8, 0.8, 2.5]
    strides, kernels = [], []

    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [
            2 if ratio <= 2 and size >= 8 else 1
            for (ratio, size) in zip(spacing_ratio, sizes)
        ]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)
    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides

if __name__ == '__main__':
    from torch import rand, randint, unbind
    import sys
    sys.path.append('..')
    from gpu_mem_track import MemTracker
    from monai.losses import  DiceCELoss

    gpu_tracker = MemTracker()
    kernels, strides = get_kernels_strides(7)
    loss_fn = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)

    model = DynUNet(opt={}).to('cuda:0')
    for i in range(2):
        dummy_input = rand((1,1,256,256,32)).cuda()
        dummy_target =randint(0,3,(1,1,256,256,32)).cuda()
        gpu_tracker.track()
        out_masks = unbind(model(dummy_input),dim=1)
        gpu_tracker.track()
        loss = sum(
                        0.5 ** i * loss_fn.forward(out, dummy_target) for i,out in enumerate(out_masks)
                    )
        gpu_tracker.track()
        gpu_tracker.clear_cache() 
        gpu_tracker.track()
    import pdb; pdb.set_trace()