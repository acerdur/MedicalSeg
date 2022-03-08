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
from util.train_util import get_model, get_optimizer, get_scheduler, get_dataloaders, prepare_for_loss
from util.logger import log
from loss import get_loss
from monai.metrics import DiceMetric
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def save_checkpoints(opt, model, optimizer, scheduler, epoch):
    model_name = type(model.module).__name__.lower()
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    save_path = os.path.join(save_dir,f"{model_name}_ckpt_{epoch}.pth")

    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        torch.save({
            'epoch': epoch,
            'state_dict': model.module.cpu().state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()},
            save_path)
        model.cuda(opt.gpu_ids[0])
    else:
        torch.save({
            'epoch': epoch,
            'state_dict': model.cpu().state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()},
            save_path)

def load_checkpoints(opt, model, optimizer, scheduler):
    model_name = type(model.module).__name__.lower()
    load_dir = os.path.join(opt.checkpoints_dir, opt.name)
    load_path = os.path.join(load_dir,f"{model_name}_ckpt_{opt.epoch_count}.pth")

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
    epoch = checkpoint['epoch']

    return model, optimizer, scheduler, epoch

def evaluate(dataloader, model, metric, opt):
    for i, data in enumerate(tqdm(dataloader)):
        scans, label_masks = data

        if not opt.no_cuda:
            scans = scans.cuda()
            label_masks.cuda()

        if not opt.baseline == 'monainet':
                scans = scans.permute(0,1,4,2,3)
                label_masks = label_masks.permute(0,3,1,2)

        with torch.no_grad():
            preds, label_masks = model.module.inference(scans, label_masks)
            metric(preds, label_masks)
    
    metric_value = metric.aggregate().item()

    return metric_value



def train(train_loader,
        val_loader, 
        model, 
        optimizer, 
        scheduler,
        metric,
        opt):
    
    batch_size = train_loader.batch_size
    batches_per_epoch = len(train_loader)
    total_epochs = opt.n_epochs + opt.n_epochs_decay
    max_num_iters = total_epochs * batches_per_epoch
    log.info(f"{total_epochs} epochs in total, {batches_per_epoch} batches per epoch.")

    if opt.continue_train:
        model, optimizer, scheduler, current_epoch = load_checkpoints(opt, model, optimizer, scheduler)
        total_iters = batches_per_epoch * current_epoch
    else:
        current_epoch = opt.epoch_count
        total_iters = 0

    log_dir = os.path.join(opt.checkpoints_dir,opt.name,'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    
    model.train()
    train_start_time = time.time()
    for epoch in range(current_epoch, total_epochs + 1):
        log.info(f"Starting epoch {epoch}")
        epoch_start_time = time.time()
        epoch_iter = 0

        log.info(f"Current lr: {scheduler.get_last_lr()}")

        for batch_id, batch_data in enumerate(train_loader):
            iter_start_time = time.time()
            scans, label_masks = batch_data

            total_iters += 1
            epoch_iter += 1

            if not opt.no_cuda:
                scans = scans.cuda()
                label_masks = label_masks.cuda()

            if not opt.baseline == 'monainet':
                scans = scans.permute(0,1,4,2,3)
                label_masks = label_masks.permute(0,3,1,2)

            # start = 0 
            # for i in range(scans.shape[-1]):
            #     if not (scans[0,:,:,:,i] == torch.zeros((1,256,256))).all():
            #         start = i
            #         break
            # end = 0
            # for i in range(start, scans.shape[-1]):
            #     if not (scans[0,:,:,:,i] == torch.zeros((1,256,256))).all():
            #         end += 1

            #import pdb; pdb.set_trace()
            optimizer.zero_grad()
            out_masks = model(scans)

            #import pdb; pdb.set_trace()
            out_masks, new_label_masks = prepare_for_loss(opt, out_masks, label_masks)
            #import pdb; pdb.set_trace()
            loss = loss_fn(out_masks, new_label_masks)
            #import pdb; pdb.set_trace()

            loss.backward()
            #import pdb; pdb.set_trace()
            optimizer.step()
            #import pdb; pdb.set_trace()

            

            if total_iters % opt.print_freq == 0:
                iter_time = (time.time() - iter_start_time)
                log.info(
                    "Iteration: {} / {} | loss = {:.3f} | iter time = {:.3f}s"\
                    .format(total_iters,max_num_iters, loss.item(), iter_time))
                writer.add_scalar("[Train] Train loss", loss.item(), total_iters)
                torch.cuda.empty_cache()
                mem = float(torch.cuda.memory_allocated() / (1024 * 1024))
                print("memory allocated:", mem, "MiB")

                # for obj in gc.get_objects():
                #     try:
                #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                #             print(type(obj), obj.size())
                #     except:
                #         pass

            if (total_iters % opt.val_freq == 0) and val_loader is not None:
                metric.reset()
                metric_value = evaluate(val_loader, model, metric, opt)
                
                log.info("Validation Dice Score: {:.4f}".format(metric_value))
                writer.add_scalar("[Val] Dice Score", metric_value, total_iters)

        #scheduler step at the end of epoch
        scheduler.step()
        if epoch % opt.save_epoch_freq == 0:
            log.info(f"Saving checkpoint at the end of epoch {epoch}, total iters {total_iters}")
            save_checkpoints(opt, model, optimizer, scheduler, epoch)

        log.info(f"End of epoch {epoch} \t Time Taken: {time.time() - epoch_start_time} sec")
    
    log.info(f"End of training \t Time Taken: {time.time() - train_start_time} sec")


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    #setup
    opt = TrainOptions().parse()
    model, parameters = get_model(opt)
    optimizer = get_optimizer(opt, parameters)
    scheduler = get_scheduler(opt, optimizer)
    loss_fn = get_loss(opt)
    train_loader, val_loader = get_dataloaders(opt)
    dice_metric = DiceMetric()

    train(train_loader, val_loader, model, optimizer, scheduler, dice_metric, opt)
