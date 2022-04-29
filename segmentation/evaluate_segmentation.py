import os 
import glob
import torch 
import time
import numpy as np
#import nibabel as nib
import SimpleITK as sitk
import torchvision
from tqdm import tqdm
#from pathlib import Path

from monai.metrics import DiceMetric
from monai.transforms import Compose, AsDiscrete, Activations#, KeepLargestConnectedComponent, FillHoles
from monai.inferers import sliding_window_inference 
from monai.data import decollate_batch

from skimage.measure import label, regionprops
from skimage.morphology import remove_small_holes, binary_erosion, binary_dilation, ball

from options.test_options import TestOptions
from util.train_util import get_model, get_dataloaders, get_pre_classifier, prepare_for_loss
from util.logger import log
from util.util import crop_to_bbox



def load_checkpoints(opt, model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model_name = type(model).__name__.lower()
    load_dir = os.path.join(opt.checkpoints_dir, opt.name)
    file_name = f"{model_name}_ckpt_{opt.epoch}.pth" if opt.epoch else "best_model.pth" if opt.use_best else "latest_model.pth" 
    load_path = os.path.join(load_dir, file_name)

    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu') 

    print(f'Loading training checkpoint from {load_path}')
    checkpoint = torch.load(load_path, map_location=device)
    if hasattr(checkpoint['state_dict'], '_metadata'):
        del checkpoint['state_dict']._metadata
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer']) # gives error
    best_score = checkpoint['best_score']
    epoch = checkpoint['epoch']

    model.eval()
    return model, epoch, best_score

def postprocessing(segmentation):
    """
    Post-processing on the segmentation outputs, e.g. keeping largest connected component & small hole removal
    They are only applied on the pancreas labels. 
    Tumor labels are then later pruned into the outputs that are inside the pancreas label
    """

    #device = segmentation.device

    # first convert one-hot encoded preds to single channel
    segmentation = segmentation.argmax(dim=0).cpu()

    # extract tumor predictions
    tumor_preds = (segmentation == 2).cpu().numpy()
    tumor_vol = tumor_preds.sum().item()
    # merge tumor & pancreas predictions
    segmentation = torch.where(segmentation != 0, 1, segmentation)
    pancreas_vol = segmentation.sum().item()

    # keep only largest connected pancreas component
    try:
        labels = label(segmentation)
        regions = regionprops(labels)
        area_sizes = []
        for region in regions:
            area_sizes.append([region.label, region.area])
        area_sizes = np.array(area_sizes)
        tmp = np.zeros_like(segmentation)
        tmp[labels == area_sizes[np.argmax(area_sizes[:, 1]), 0]] = 1
        #segmentation = tmp.copy()
        labels = None
        regions = None 
        area_sizes = None
    except Exception as e:
        print(e)
        #import pdb; pdb.set_trace()

    # dilation 
    #tmp = binary_dilation(tmp.astype(bool), ball(3))
    #tumor_preds = binary_dilation(tumor_preds, ball(3))

    # remove small holes
    tmp = remove_small_holes(
        tmp.astype(bool), area_threshold=0.001 * np.prod(tmp.shape)
    ).astype(np.float32)

    # filter tumor preds by final pancreas area
    tumor_preds = np.where(tmp == 1, tumor_preds, False)

    tumor_vol_new = tumor_preds.sum().item()
    pancreas_vol_new = tmp.sum().item()

    # re-inject filtered tumor preds to the final preds
    tmp = np.where(tumor_preds, 2, tmp)
    tumor_preds = None

    # turn back to torch & one-hot encoding for DiceMetric
    segmentation = AsDiscrete(to_onehot=3)(torch.Tensor(tmp).unsqueeze(0))
    tmp = None

    #segmentation = segmentation.to(device)

    #print(f"Pancreas volume is reduced from {pancreas_vol} to {pancreas_vol_new}")
    #print(f"Tumor volume is reduced from {tumor_vol} to {tumor_vol_new}")

    return segmentation

    
def evaluate(dataloader, model, preclassifier, metrics, post_trans_pred, post_trans_lbl, opt):
    all_preds = {}
    all_preds_processed = {}
    with torch.inference_mode():
        for i, data in enumerate(tqdm(dataloader)):
            scans, label_masks, scan_paths = data # B x 1 x H x W x D scans ; B x H x W x D label_masks

            if not opt.no_cuda:
                scans = scans.cuda()
                label_masks = label_masks.cuda()

            cropped_top = 0
            cropped_bottom = 0

            if not opt.no_pre_cropping:
                slice_crop_mask, window_indices = preclassifier.soft_pred(scans.permute(0,4,1,2,3), window_size_mult_of=opt.wind_size_mult_of)
                slice_crop_mask = slice_crop_mask.to(bool)
                #useful for padding while saving preds later on
                cropped_top += window_indices[0]
                cropped_bottom = scans.shape[-1] - window_indices[1]
                #import pdb; pdb.set_trace()
                scans = scans[:,:,:,:,slice_crop_mask]#, label_masks[:,:,:,slice_crop_mask]
            
                
            if not opt.baseline == 'monainet':
                scans = scans.permute(0,1,4,2,3) # B x 1 x D x H x W 
                label_masks = label_masks.permute(0,3,1,2)  # B x D x H x W 

            if opt.baseline == 'lstm2d':
                scans = scans.permute(0,2,1,3,4) # B x D x 1 x H x W
            
            preds, label_masks = model.inference(scans, label_masks)
            
            #import pdb; pdb.set_trace()
            #preds = [post_trans_pred(pred) for pred in decollate_batch(preds)]
            preds = preds.argmax(dim=1,keepdim=True)
           
            preds = torch.nn.functional.pad(preds,(cropped_top, cropped_bottom),mode='constant',value=0)
            preds = [post_trans_lbl(pred) for pred in decollate_batch(preds)]
            label_masks = [post_trans_lbl(label_mask) for label_mask in decollate_batch(label_masks)]
            metrics[0](preds, label_masks)
            all_preds[scan_paths[0]] = preds[0]

            #import pdb; pdb.set_trace()

            if not opt.no_postprocessing:
                preds_processed = [postprocessing(pred) for pred in preds]
                label_masks = [lbl.cpu() for lbl in label_masks]
                metrics[1](preds_processed, label_masks)
                all_preds_processed[f"{scan_paths[0]}_post"] = preds_processed[0]

            #scans = crop_to_bbox(scans, preds_processed[0], 10)

    return metrics , all_preds, all_preds_processed

def save_to_nifti(preds, results_dir, opt):
    """
    Resize not implemented yet
    preds:  Dict of predictions provided by evaluate()
            Items should have {scan_path: predictions} form
    """
    #import pdb; pdb.set_trace()
    original_shape = (512,512)

    for scan_pth, pred in preds.items():
        #original_scan = nib.load(scan_pth.split('_')[0])
        original_scan = sitk.ReadImage(scan_pth.split('_')[0])
        name = scan_pth.split('/')[-1].split('.')[0] 
        name = name + "_post" if "_post" in scan_pth else name
        dataset_name = scan_pth.split('/')[-3]
        pred = pred.argmax(dim=0) # to H x W x D
        pred = torchvision.transforms.functional.resize(pred.permute(2,0,1), (original_shape), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        pred_numpy = pred.permute(1,2,0).cpu().numpy().astype(np.int8) 
        pred_numpy = np.rot90(np.fliplr(pred_numpy), k=3) # back transformation from refrence position to original scan coordinates
        #pred_nifti = nib.Nifti1Image(pred_numpy,affine=original_scan.affine)
        pred_nifti = sitk.GetImageFromArray(np.swapaxes(pred_numpy, 0, 2))

        os.makedirs(os.path.join(results_dir, dataset_name), exist_ok=True)
        sitk.WriteImage(pred_nifti, os.path.join(results_dir, dataset_name, f"{name}.nii"))
        #nib.save(pred_nifti, os.path.join(results_dir, dataset_name, name))

if __name__ == '__main__':
    opt = TestOptions().parse()
    
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
    dice_metric_raw = DiceMetric(reduction='mean_batch')
    dice_metric_raw.__name__ = 'Dice Score on Raw Predictions'
    metrics = [dice_metric_raw]
    if not opt.no_postprocessing:
        dice_metric_post = DiceMetric(reduction='mean_batch')
        dice_metric_post.__name__ = 'Dice Score on Post-processed Predictions'
        metrics.append(dice_metric_post)

    post_trans_pred = Compose([AsDiscrete(argmax=True, to_onehot=3)]) #[Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_trans_lbl = Compose([AsDiscrete(to_onehot=3)])

    metrics, preds, preds_processed = evaluate(val_loader, model, pre_classifier, metrics, post_trans_pred, post_trans_lbl, opt)
    for metric in metrics:
        print(metric.__name__)
        metric_per_class = metric.aggregate().flatten() 
        mean_metric = metric_per_class.mean().item() 
        [log.info("Val Dice Class {}: {:.4f}".format(i,score.item())) for i,score in enumerate(metric_per_class)]
        log.info("Val Mean Dice: {:.4f}".format(mean_metric))

    if opt.save_results:
        results_dir = os.path.join('./results', opt.name)
        os.makedirs(results_dir, exist_ok=True)
    
    if opt.save_results == 'raw':
        save_to_nifti(preds, results_dir, opt)
    elif opt.save_results == 'processed':
        save_to_nifti(preds_processed, results_dir, opt)
    elif opt.save_results == 'all':
        save_to_nifti(preds.update(preds_processed), results_dir, opt)
    else:
        pass
    