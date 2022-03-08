from torch.nn import CrossEntropyLoss
from monai.losses import DiceLoss, DiceCELoss

def get_loss(opt):

    if opt.loss_function == 'cross_entropy':
        loss_fn = CrossEntropyLoss()
    elif opt.loss_function == 'dice':
        loss_fn = DiceLoss(to_onehot_y=True, sigmoid=True)
    elif opt.loss_function == 'dice_cross_entropy':
        loss_fn = DiceCELoss(to_onehot_y=True, sigmoid=True)
    else:
        raise NotImplementedError('Specified loss_function {} is not known.'.format(opt.loss_function))

    if not opt.no_cuda:
        loss_fn = loss_fn.cuda()

    return loss_fn