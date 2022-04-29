import argparse
import os
from util import util
import torch
#import models
#import data


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.

    The model options are sensitive for the selected model. 
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        subparsers = parser.add_subparsers(help='sub-commands for different models', dest='baseline')
        medicalnet = subparsers.add_parser('medicalnet', help='use MedicalNet ResNet')
        monainet = subparsers.add_parser('monainet', help='use MonAI networks ')
        lstm2d = subparsers.add_parser('lstm2d', help='use LSTM based 2.5D segmentation net. Only available for eval')

        # basic parameters
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        
        # model parameters
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization ( normal | xavier | kaiming | orthogonal )')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--memory_format', type=str, default='channels_first', help=' ( channels_first, channels_last ) ')
            # parameters only for MedicalNet ResNet
        medicalnet.add_argument('--model_depth', default=50, type=int, help='Depth of resnet ( 10 | 18 | 34 | 50 | 101 | 152 | 200 )')
        medicalnet.add_argument('--resnet_shortcut', default='B', type=str, help='Shortcut type of resnet ( A | B )')
        medicalnet.add_argument('--input_D',default=56, type=int, help='Input size of depth')
        medicalnet.add_argument('--input_H', default=256, type=int, help='Input size of height')
        medicalnet.add_argument('--input_W', default=256, type=int, help='Input size of width')
        medicalnet.add_argument('--num_seg_classes', default=3, type=int, help='Number of segmentation classes')
            # MonAI networks parameters
        monainet.add_argument('--model', default='UNet', type=str, help='Which MonAI network, Uppercase sensitive ( UNet | SegResNet | VNet)')
            # LSTM 2.5D parameters
        lstm2d.add_argument('--ckpt_folder', type=str, help='Folder that containss the trained checkpoint to load LSTM 2.5D model')
        
        # dataset parameters
        parser.add_argument('--data_config', required=True, help='path to .yaml file which contains data configuration')
        parser.add_argument('--num_workers', default=2, type=int, help='number of workers for loading data')
        # additional parameters
        parser.add_argument('--epoch', type=str, default=None, help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # # modify model-related parser options
        # model_name = opt.model
        # model_option_setter = models.get_option_setter(model_name)
        # parser = model_option_setter(parser, self.isTrain)
        # opt, _ = parser.parse_known_args()  # parse again with new defaults

        # # modify dataset-related parser options
        # dataset_name = opt.dataset_mode
        # dataset_option_setter = data.get_option_setter(dataset_name)
        # parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])
            opt.no_cuda = False
        else:
            opt.no_cuda = True
        
        opt.inference_mode = False
        
        self.opt = opt
        return self.opt
