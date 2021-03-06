from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # transfer learning, pretrained networks - different than training continue
        parser.add_argument('--pretrain_path', type=str, default='', help='Path to .pth file containing pretrained model.')
        parser.add_argument('--new_layer_names', default=['all'], type=list, help='Unfrozen layers, that will have the specified learning rate. lr for other layers will be divided by 100. Set ["all"] to consider all layers as new.')

        # 2D pre-classification model parameters
        parser.add_argument('--no_pre_cropping', action='store_true', help='Do not use 2.5D LSTM classifier to crop scans into a ROI containing pancreas')
        parser.add_argument('--classifier_weight_folder', type=str, default="/home/erdurc/punkreas/segmentation/pretraining_2D/models/model_ckpt_deeplabv3plus_resnet50_classification_lstm_2022-05-10_15-22", help='Path to the file containing trained weights of 2D pre-classifier.')
        parser.add_argument('--wind_size_mult_of', type=int, default=16, help='The prediction window of 2D pre-classifier should be a multiple of this.')
        
        # network saving and loading parameters
        #parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--print_freq', type=int, default=10, help='iteration frequency of printing progress')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=60, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=60, help='number of epochs to linearly decay learning rate to zero (linear policy)')
        parser.add_argument('--loss_function', type=str, default='dice_cross_entropy', help='Loss function to use ( cross_entropy | dice | dice_cross_entropy')
        parser.add_argument('--optimizer', type=str, default='AdamW', help='Which optimizer to use ( SGD | Adam | AdamW )')
        parser.add_argument('--betas', nargs='+' ,type=float, default=[0.9, 0.99], help='Momentum values for optimizer.')
        parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for new parameters')
        parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for optimizers')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations (step policy)')
        parser.add_argument('--val_freq', type=int, default=250, help='Validation frequency by iteration')
        parser.add_argument('--patience', type=int, default=15, help='Patience for early stopping. Counted after validation step.')
        self.isTrain = True
        return parser
