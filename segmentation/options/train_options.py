from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # transfer learning, pretrained networks - different than training continue
        parser.add_argument('--pretrain_path', type=str, default='', help='Path to .pth file containing pretrained model.')
        parser.add_argument('--new_layer_names', default=['conv_seg'], type=list, help='Unfrozen layers, that will have the specified learning rate. lr for other layers will be divided by 100.')

        # 2D pre-classification model parameters
        parser.add_argument('--classifier_type', type=str, default='lstm', help='Type of the 2D pre-classifier. [ lstm | linear ]')
        parser.add_argument('--classifier_weights', type=str, help='Path to the file containing trained weights of 2D pre-classifier.')
        
        # network saving and loading parameters
        #parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--print_freq', type=int, default=10, help='iteration frequency of printing progress')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero (linear policy)')
        parser.add_argument('--loss_function', type=str, default='dice_cross_entropy', help='Loss function to use ( cross_entropy | dice | dice_cross_entropy')
        parser.add_argument('--optimizer', type=str, default='Adam', help='Which optimizer to use ( SGD | Adam | AdamW )')
        parser.add_argument('--betas', nargs='+' ,type=float, default=[0.9, 0.99], help='Momentum values for optimizer.')
        parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate for new parameters')
        parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay for optimizers')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations (step policy)')
        parser.add_argument('--val_freq', type=int, default=250, help='Validation frequency')
        self.isTrain = True
        return parser
