import os 
import glob
import torch 
import yaml
import time
import numpy as np
import random
import gc

#from punkreas import data, transform, optimizer
from options.train_options import TrainOptions
from util.train_util import get_model, get_optimizer, get_scheduler, get_dataloaders, get_pre_classifier, prepare_for_loss
from util.logger import log
from loss import get_loss

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from monai.metrics import DiceMetric
from monai.transforms import Compose, AsDiscrete, Activations
from monai.inferers import sliding_window_inference 
from monai.data import decollate_batch

from gpu_mem_track import MemTracker

def save_checkpoints(opt, model, optimizer, scheduler, epoch, best_score, monitoring, filename=''):
    model_name = type(model.module).__name__.lower()
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    if not filename:
        save_path = os.path.join(save_dir,f"{model_name}_ckpt_{epoch}.pth")
    else: 
        save_path = os.path.join(save_dir, filename)

    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        torch.save({
            'epoch': epoch,
            'best_score': best_score,
            'monitoring_state': monitoring,
            'state_dict': model.module.cpu().state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()},
            save_path)
        model.cuda(opt.gpu_ids[0])
    else:
        torch.save({
            'epoch': epoch,
            'best_score': best_score,
            'monitoring_state': monitoring,
            'state_dict': model.cpu().state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()},
            save_path)

def load_checkpoints(opt, model, optimizer, scheduler,):
    model_name = type(model.module).__name__.lower()
    load_dir = os.path.join(opt.checkpoints_dir, opt.name)
    if opt.epoch:
        load_path = os.path.join(load_dir,f"{model_name}_ckpt_{opt.epoch}.pth")
    else: 
        load_path = os.path.join(load_dir, 'best_model.pth')
    
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu') 
    
    
    print(f'Loading training checkpoint from {load_path}')
    checkpoint = torch.load(load_path)#, map_location=device)
    if hasattr(checkpoint['state_dict'], '_metadata'):
        del checkpoint['state_dict']._metadata
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer']) # gives error
    scheduler.load_state_dict(checkpoint['scheduler'])
    best_score = checkpoint['best_score']
    epoch = checkpoint['epoch']
    monitoring = checkpoint['monitoring_state']

    scheduler.step() #update lr from the loaded checkpoint

    return model, optimizer, scheduler, epoch, best_score, monitoring

def evaluate(dataloader, model, preclassifier, metric, post_trans_pred, post_trans_lbl, opt):
    with torch.inference_mode():
        for i, data in enumerate(tqdm(dataloader)):
            scans, label_masks, scan_paths, extra_info = data
            # if 'CropToMask' in extra_info:
            #     cropped_top = extra_info['CropToMask'][-1][0].item()
            #     cropped_bottom = extra_info['CropToMask'][-1][1].item()
            # else:
            #     cropped_top=0
            #     cropped_bottom=0

            if not opt.no_cuda:
                scans = scans.cuda()
                label_masks = label_masks.cuda()
        
            # slice_crop_mask, _ = preclassifier.soft_pred(scans.permute(0,4,1,2,3), window_size_mult_of=opt.wind_size_mult_of)
            # slice_crop_mask = slice_crop_mask.to(bool)
            # scans, label_masks = scans[:,:,:,:,slice_crop_mask], label_masks[:,:,:,slice_crop_mask]
            
            if not opt.baseline == 'monainet':
                scans = scans.permute(0,1,4,2,3)
                label_masks = label_masks.permute(0,3,1,2)
            
            preds, label_masks = model.module.inference(scans, label_masks)
            #preds = preds.argmax(dim=1,keepdim=True)
            #preds = torch.nn.functional.pad(preds,(cropped_top, cropped_bottom),mode='constant',value=0)
            preds = [post_trans_pred(pred) for pred in decollate_batch(preds)]
            label_masks = [post_trans_lbl(label_mask) for label_mask in decollate_batch(label_masks)]
            metric(preds, label_masks)

            del scans 
            del preds 
            del label_masks
                
        metric_per_class = metric.aggregate().flatten()
        mean_metric= metric_per_class[1:].mean().item()
        metric.reset()
    
    return metric_per_class, mean_metric



def train(train_loader,
        val_loader, 
        model, 
        pre_classifier,
        optimizer, 
        scheduler,
        metric,
        opt):
    
    batch_size = train_loader.batch_size
    batches_per_epoch = len(train_loader)
    total_epochs = opt.n_epochs + opt.n_epochs_decay
    max_num_iters = total_epochs * batches_per_epoch
    log.info(f"{total_epochs} epochs in total, {batches_per_epoch} batches per epoch.")

    post_trans_pred = Compose([AsDiscrete(argmax=True, to_onehot=3)]) #[Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_trans_lbl = Compose([AsDiscrete(to_onehot=3)])

    if opt.continue_train:
        model, optimizer, scheduler, last_epoch, best_score, monitoring = load_checkpoints(opt, model, optimizer, scheduler)
        total_iters = batches_per_epoch * last_epoch
        current_epoch = last_epoch + 1
    else:
        current_epoch = 1
        total_iters = 0
        best_score = 0.0
        monitoring = 0

    import pdb; pdb.set_trace()
    break_training = False 

    log_dir = os.path.join(opt.checkpoints_dir,opt.name,'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    
    #gpu_tracker = MemTracker() 
    model.train()
    train_start_time = time.time()
    for epoch in range(current_epoch, total_epochs + 1):
        log.info(f"Starting epoch {epoch}")
        epoch_start_time = time.time()
        epoch_iter = 0

        log.info(f"Current lr: {scheduler.get_last_lr()}")

        for batch_id, batch_data in enumerate(train_loader):
            iter_start_time = time.time()
            scans, label_masks, scan_paths, _ = batch_data
            # when using CropToMask with lstm_classifier outputs; scan and mask are loaded cropped
            
            total_iters += 1
            epoch_iter += 1

            if not opt.no_cuda:
                scans = scans.cuda()
                label_masks = label_masks.cuda()

            #import pdb; pdb.set_trace()
            # MOVED into dataset preprocessing
            # with torch.no_grad():
            #     slice_crop_mask, _ = pre_classifier.soft_pred(scans.permute(0,4,1,2,3), window_size_mult_of=opt.wind_size_mult_of)
            #     slice_crop_mask = slice_crop_mask.to(bool)           
            # #slice_lbl = torch.any(torch.any(label_masks,dim=1),dim=1).to(int)

            # scans = scans[:,:,:,:,slice_crop_mask]
            # label_masks = label_masks[:,:,:,slice_crop_mask]

            if (scans.shape[-1] % opt.wind_size_mult_of) != 0:
            # in some rare cases the window takes the whole scan, 
            # which has a shape incompatible with MonAI nets (e.g. 44 when it should be 48)
            # apply a small 0 padding trick for compatibility
                size_diff = int(round(scans.shape[-1] / opt.wind_size_mult_of) * opt.wind_size_mult_of) - scans.shape[-1]
                scans = torch.nn.functional.pad(scans,(0,size_diff))
                label_masks = torch.nn.functional.pad(label_masks,(0,size_diff))
            
            if not opt.baseline == 'monainet':
                scans = scans.permute(0,1,4,2,3)
                label_masks = label_masks.permute(0,3,1,2)

            optimizer.zero_grad()
            try:
                out_masks = model(scans)
            except Exception as e:
                print(e)
                import pdb; pdb.set_trace()

            #gpu_tracker.track()
            out_masks, label_masks = prepare_for_loss(opt, out_masks, label_masks)
            #import pdb; pdb.set_trace()
            if len(out_masks.shape) == 6:
                ## deep supervision mode for DynUNet
                out_masks = torch.unbind(out_masks, dim=1)
                loss = sum(
                    0.5 ** i * loss_fn.forward(out, label_masks) for i,out in enumerate(out_masks)
                )
            else:
                loss = loss_fn(out_masks, label_masks)
            #import pdb; pdb.set_trace()
        
            del scans
            del out_masks

            loss.backward()
            #import pdb; pdb.set_trace()
            optimizer.step()
            #import pdb; pdb.set_trace()
            #gpu_tracker.track()

            if (total_iters == 0) or (total_iters % opt.print_freq == 0):
                iter_time = (time.time() - iter_start_time)
                log.info(
                    "Iteration: {} / {} | loss = {:.3f} | iter time = {:.3f}s"\
                    .format(total_iters,max_num_iters, loss.float(), iter_time))
                writer.add_scalar("[Train] Train loss", loss.float(), total_iters)
                torch.cuda.empty_cache()
                #mem = float(torch.cuda.memory_allocated() / (1024 * 1024))
                #res = float(torch.cuda.memory_reserved() / (1024 * 1024))
                #print("memory allocated:", mem, "MiB")
                #print("memory reserved:", res, "MiB")
                #print(torch.cuda.memory_summary())

            del loss

            if (total_iters % opt.val_freq == 0) and val_loader is not None:
                model.eval()
                metric_per_class, mean_metric = evaluate(val_loader, model, pre_classifier, metric, post_trans_pred, post_trans_lbl, opt)
                model.train()

                [log.info("Val Dice Class {}: {:.4f}".format(i,score.item())) for i,score in enumerate(metric_per_class)]
                log.info("Val Mean Dice: {:.4f}".format(mean_metric))
                writer.add_scalar("[Val] Mean Dice", mean_metric, total_iters)
                [writer.add_scalar(f"[Val] Dice Class[{i}]",score.item(), total_iters) for i,score in enumerate(metric_per_class)]

                if mean_metric > best_score:
                    best_score = mean_metric
                    monitoring = 0
                    save_checkpoints(opt, model, optimizer, scheduler, epoch, best_score, monitoring, filename='best_model.pth')

                else:
                    monitoring += 1
                    log.info(f"Val score not improved. Current status on early stopping: {monitoring} / {opt.patience}")                  
                    if monitoring >= opt.patience:
                        log.info(f"Patience for early stopping is reached. Finishing training")
                        break_training = True
                        break

        if break_training:
            #save_checkpoints(opt, model, optimizer, scheduler, epoch, best_score, filename='latest_model.pth')
            break 
        #scheduler step at the end of epoch
        scheduler.step()
        if epoch % opt.save_epoch_freq == 0:
            log.info(f"Saving checkpoint at the end of epoch {epoch}, total iters {total_iters}")
            save_checkpoints(opt, model, optimizer, scheduler, epoch, best_score, monitoring)

        log.info(f"End of epoch {epoch} \t Time Taken: {time.time() - epoch_start_time} sec")
    
    save_checkpoints(opt, model, optimizer, scheduler, epoch, best_score, monitoring, filename='latest_model.pth')
    log.info(f"End of training \t Time Taken: {time.time() - train_start_time} sec")
    

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    #setup
    opt = TrainOptions().parse()
    model, parameters = get_model(opt)
    pre_classifier = None #get_pre_classifier(opt)
    optimizer = get_optimizer(opt, parameters)
    scheduler = get_scheduler(opt, optimizer)
    loss_fn = get_loss(opt)
    train_loader, val_loader = get_dataloaders(opt)
    dice_metric = DiceMetric(reduction='mean_batch')

    train(train_loader, val_loader, model, pre_classifier, optimizer, scheduler, dice_metric, opt)
