import os 
import glob
import torch 
from tqdm import tqdm
from util.train_util import get_dataloaders, get_pre_classifier, prepare_for_loss

from options.test_options import TestOptions
from evaluate_segmentation import save_to_nifti

from monai.transforms import Compose, AsDiscrete, Activations
from monai.inferers import sliding_window_inference 
from monai.data import decollate_batch


def find_slice_masks(dataloader,classifier,opt):
    to_onehot = Compose([AsDiscrete(to_onehot=3)])

    pred_dict = {}
    with torch.inference_mode():
        for i,batch in enumerate(tqdm(dataloader)):
            scan, mask, scan_path = batch
            if not opt.no_cuda:
                scan = scan.cuda()
            
            #import pdb; pdb.set_trace()
            slice_crop_mask, _ = classifier.soft_pred(scan.permute(0,4,1,2,3), window_size_mult_of=opt.wind_size_mult_of)
            slice_crop_mask = slice_crop_mask.to(bool)
            preds = torch.zeros_like(scan).cpu()
            preds[:,:,:,:,slice_crop_mask] = 1
            preds = [to_onehot(pred) for pred in decollate_batch(preds)]
            pred_dict[scan_path[0]] = preds[0]

    return pred_dict



if __name__ == "__main__":
    opt = TestOptions().parse()
    opt.inference_mode = True
    opt.name = "lstm2d_classifier"
    os.makedirs(os.path.join(opt.checkpoints_dir,opt.name),exist_ok=True)

    slice_classifier = get_pre_classifier(opt)
    train_loader, val_loader = get_dataloaders(opt)

    pred_dict = find_slice_masks(train_loader,slice_classifier,opt)
    pred_dict.update(find_slice_masks(val_loader,slice_classifier,opt))

    if opt.save_results:
        results_dir = os.path.join('./results', opt.name)
        os.makedirs(results_dir, exist_ok=True)

        save_to_nifti(pred_dict, results_dir, opt)

    


