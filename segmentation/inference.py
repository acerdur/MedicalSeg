import os
import torch
import gc
from datetime import datetime
from tqdm import tqdm

from monai.transforms import Compose, AsDiscrete, Activations
from monai.inferers import sliding_window_inference 
from monai.data import decollate_batch

from options.test_options import TestOptions
from util.train_util import get_model, get_dataloaders, get_pre_classifier, prepare_for_loss
from evaluate_segmentation import save_to_nifti, load_checkpoints, postprocessing


def main(loader, model, preclassifier, opt):

    #post_trans_pred = Compose([AsDiscrete(argmax=True, to_onehot=3)]) #[Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    to_onehot = Compose([AsDiscrete(to_onehot=3)])

    opt.name = "lstm2d_classifier"
    if opt.save_results:
        results_dir = os.path.join('./results', opt.name)
        os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    with torch.inference_mode():
        for i, data in enumerate(tqdm(loader)):
            scans, _ , scan_paths = data # B x 1 x H x W x D scans ; B x H x W x D 
            _ = None

            #import pdb; pdb.set_trace()

            if not opt.no_cuda:
                scans = scans.cuda()

            cropped_top = 0
            cropped_bottom = 0

            
            if not opt.no_pre_cropping:
                try:
                    slice_crop_mask, window_indices = preclassifier.soft_pred(scans.permute(0,4,1,2,3), window_size_mult_of=opt.wind_size_mult_of)
                    slice_crop_mask = slice_crop_mask.to(bool)
                    #useful for padding while saving preds later on
                    cropped_top = 0 + window_indices[0]
                    cropped_bottom = scans.shape[-1] - window_indices[1]
                    preds = torch.zeros_like(scans)
                    preds[:,:,:,:,slice_crop_mask] = 1
                    scans = scans[:,:,:,:,slice_crop_mask]
                except:
                    print(f"LSTM classifier couldn't find pancreas on scan: {scan_paths[0].split('/')[-1]}")
                    with open(os.path.join(results_dir, f'{timestamp}_failed_scans.txt'), 'a') as f:
                        f.write(f"{scan_paths[0]}\tError source: LSTM classifier \n")
                    continue 

            if not opt.baseline == 'monainet':
                scans = scans.permute(0,1,4,2,3) # B x 1 x D x H x W 

            try:
                # if opt.baseline == 'lstm2d':
                #     scans = scans.permute(0,2,1,3,4) # B x D x 1 x H x W
                #     _, preds = model(scans)
                #     preds = preds.permute(0,2,3,4,1) # to B x 3 x H x W x D
                # else:
                #     preds = model(scans)
                   
                # scans = None 
                # slice_crop_mask = None

                # preds = preds.argmax(dim=1,keepdim=True)
                # preds = torch.nn.functional.pad(preds,(cropped_top, cropped_bottom),mode='constant',value=0)
                preds = [to_onehot(pred) for pred in decollate_batch(preds)]
                pred_dict = {scan_paths[0]: preds[0]}
                
                if not opt.no_postprocessing:
                    preds = [postprocessing(pred) for pred in preds]
                    pred_dict_post = {f"{scan_paths[0]}_post": preds[0]}
            except Exception as e:
                print(f"3D Model couldn't find pancreas/tumor on scan: {scan_paths[0].split('/')[-1]}")
                print(e)
                with open(os.path.join(results_dir, f'{timestamp}_failed_scans.txt'), 'a') as f:
                    f.write(f"{scan_paths[0]}\tError source: 3D model \n")
                continue

            
            if opt.save_results == 'raw':
                save_to_nifti(pred_dict, results_dir, opt)
            elif opt.save_results == 'processed':
                save_to_nifti(pred_dict_post, results_dir, opt)
            elif opt.save_results == 'all':
                save_to_nifti(pred_dict.update(pred_dict_post), results_dir, opt)
            else:
                pass

            del preds, pred_dict
            gc.collect()
            torch.cuda.empty_cache()
            


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.inference_mode = True
    model, _ = get_model(opt)
    if not opt.baseline == 'lstm2d':
        model, _ , _ = load_checkpoints(opt, model)
    else:
        opt.no_pre_cropping = True
    if not opt.no_pre_cropping:
        pre_classifier = get_pre_classifier(opt)
    else:
        pre_classifier = None
    _, val_loader = get_dataloaders(opt)

    main(val_loader, model, pre_classifier, opt)