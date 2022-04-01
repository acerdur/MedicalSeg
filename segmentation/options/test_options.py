from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--pretrain_path', default=None)
        # 2D pre-classification model parameters
        parser.add_argument('--no_pre_cropping', action='store_true', help='Do not use 2.5D LSTM classifier to crop scans into a ROI containing pancreas')
        parser.add_argument('--classifier_weight_folder', type=str, help='Path to the file containing trained weights of 2D pre-classifier.')
        parser.add_argument('--wind_size_mult_of', type=int, default=16, help='The prediction window of 2D pre-classifier should be a multiple of this.')
        
      
        #parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        
        parser.add_argument('--use_best', action='store_true', help='using the best model or the model specified by epoch')
        parser.add_argument('--use_sliding_window', action='store_true', help='use the MonAI sliding_window_inference on whole scan for inference')
        parser.add_argument('--no_postprocessing', action='store_true', help="don't apply morphological postprocessing to the predictions")       
        
        parser.add_argument('--save_results', type=str, default='', help="which results to save as .nii files, select from ( '', raw, processed, all")
        self.isTrain = False
        return parser
