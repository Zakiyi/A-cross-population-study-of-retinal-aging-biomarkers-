# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional
import numpy as np
import torch
import torch.nn as nn
from utils import AverageMeter
from timm.data import Mixup
from timm.utils import accuracy
import umap
import umap.plot as up
import matplotlib.pyplot as plt
import models.util.misc as misc
import models.util.lr_sched as lr_sched
START_AGE = 0
END_AGE = 77


class Metrics_logger:
    def __init__(self):
        self.overall_loss = AverageMeter()
        self.loss_stage1_meter = AverageMeter()
        self.loss_stage2_meter = AverageMeter()
        self.loss_align_meter = AverageMeter()
        self.loss_ordinal_meter = AverageMeter()

        self.coarse_mae = AverageMeter()
        self.refined_mae = AverageMeter()

    def update_metrics(self, size, coarse_mae, refined_mae, loss, loss_stage1, loss_stage2, loss_ordinal, loss_align):
        self.overall_loss.update(loss, size)

        self.loss_stage1_meter.update(loss_stage1, size)
        self.loss_stage2_meter.update(loss_stage2, size)
        self.loss_ordinal_meter.update(loss_ordinal, size)
        self.loss_align_meter.update(loss_align, size)

        self.coarse_mae.update(coarse_mae, size)
        self.refined_mae.update(refined_mae, size)


def train_one_epoch(model: torch.nn.Module, criterion: dict, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):

    model.train(True)
    metric_logger_wandb = Metrics_logger()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets, weight, __, ___) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs1, outputs2, feat_, feats, loss_ordinal = model(samples)

            loss_stage1 = criterion['loss_coarse'](outputs1, targets.round()) + criterion['ce_loss'](outputs1, torch.div(targets, args.age_bin, rounding_mode='floor').long())
            loss_stage2 = criterion['loss_refined'](outputs2, targets.round()) + criterion['ce_loss'](outputs2, targets.round().long())
            loss_ordinal = criterion['ordinal_loss'](feat_, targets.round())
            loss_align = criterion['align_loss'](feats[0], targets.round()) + criterion['align_loss'](feats[1], targets.round()) 
            
            if 'base' in args.ablation:
                loss = loss_stage1 + loss_stage2
            elif 'ordinal' in args.ablation:
                loss = loss_stage1 + loss_stage2 + loss_ordinal
            elif 'align' in args.ablation:
                loss = loss_stage1 + loss_stage2 + loss_align
            else:
                loss = loss_stage1 + loss_stage2 + loss_ordinal + loss_align

        loss_value = loss.item()
        loss_coarse_value = loss_stage1.item()
        loss_refined_value = loss_stage2.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        age_span = torch.arange(START_AGE, END_AGE, step=args.age_bin, dtype=torch.float32).to(samples.device)
        coarse_pred = (nn.Softmax(dim=1)(outputs1) * age_span).sum(1)  # N

        age_span = torch.arange(START_AGE, END_AGE, step=1, dtype=torch.float32).to(samples.device)
        final_pred = (nn.Softmax(dim=1)(outputs2) * age_span).sum(1)  # N

        coarse_mae = torch.abs(targets.squeeze() - coarse_pred.squeeze()).mean()
        refined_mae = torch.abs(targets.squeeze() - final_pred.squeeze()).mean()

        metric_logger_wandb.update_metrics(samples.size(0), coarse_mae.item(), refined_mae.item(), loss_value,
                                           loss_coarse_value, loss_refined_value, loss_ordinal.item(), loss_align.item())

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_coarse=loss_coarse_value)
        metric_logger.update(loss_refined=loss_refined_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    disp_str = 'Epoch {} training losses: {:.4f}, stage_1 loss: {:.4f}, stage_2 loss: {:.4f}, ordinal loss:{:.4f}, align loss:{:.4f}'.format(epoch, metric_logger_wandb.overall_loss.avg, 
    metric_logger_wandb.loss_stage1_meter.avg, metric_logger_wandb.loss_stage2_meter.avg, metric_logger_wandb.loss_ordinal_meter.avg, metric_logger_wandb.loss_align_meter.avg)
    print('\n{}'.format(disp_str), 'train mae: {:.4f}'.format(metric_logger_wandb.refined_mae.avg))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, metric_logger_wandb


@torch.no_grad()
def evaluate(args, data_loader, model, criterion, device, epoch):

    metric_logger_wandb = Metrics_logger()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for samples, targets, weight, __, ___ in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            outputs1, outputs2, feat_, feats, loss_ordinal = model(samples)
            loss_stage1 = criterion['loss_coarse'](outputs1, targets.round()) + criterion['ce_loss'](outputs1, torch.div(targets, args.age_bin, rounding_mode='floor').long())
            loss_stage2 = criterion['loss_refined'](outputs2, targets.round()) + criterion['ce_loss'](outputs2, targets.round().long())
            loss_ordinal = criterion['ordinal_loss'](feat_, targets.round())
            loss_align = criterion['align_loss'](feats[0], targets.round()) + criterion['align_loss'](feats[1], targets.round()) 
            
            if 'base' in args.ablation:
                loss = loss_stage1 + loss_stage2
            elif 'ordinal' in args.ablation:
                loss = loss_stage1 + loss_stage2 + loss_ordinal
            elif 'align' in args.ablation:
                loss = loss_stage1 + loss_stage2 + loss_align
            else:
                loss = loss_stage1 + loss_stage2 + loss_ordinal + loss_align

        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_coarse=loss_stage1.item())
        metric_logger.update(loss_refined=loss_stage2.item())

        age_span = torch.arange(START_AGE, END_AGE, step=args.age_bin, dtype=torch.float32).to(samples.device)
        coarse_pred = (nn.Softmax(dim=1)(outputs1) * age_span).sum(1)  # N

        age_span = torch.arange(START_AGE, END_AGE, step=1, dtype=torch.float32).to(samples.device)
        final_pred = (nn.Softmax(dim=1)(outputs2) * age_span).sum(1)  # N

        coarse_mae = torch.abs(targets.squeeze() - coarse_pred.squeeze()).mean()
        refined_mae = torch.abs(targets.squeeze() - final_pred.squeeze()).mean()

        metric_logger_wandb.update_metrics(samples.size(0), coarse_mae.item(), refined_mae.item(), loss.item(),
                                           loss_stage1.item(), loss_stage2.item(), loss_ordinal.item(), loss_align.item())

        metric_logger.meters['mae_coarse'].update(coarse_mae.item(), n=samples.size(0))
        metric_logger.meters['mae_refined'].update(refined_mae.item(), n=samples.size(0))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* coarse mae: {mae1.global_avg:.3f} refined_mae: {mae2.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(mae1=metric_logger.mae_coarse, mae2=metric_logger.mae_refined, losses=metric_logger.loss))
    
    disp_str = 'Epoch {} val losses: {:.4f}, stage_1 loss: {:.4f}, stage_2 loss: {:.4f}, ordinal loss:{:.4f}, align loss:{:.4f}'.format(epoch, metric_logger_wandb.overall_loss.avg, 
    metric_logger_wandb.loss_stage1_meter.avg, metric_logger_wandb.loss_stage2_meter.avg, metric_logger_wandb.loss_ordinal_meter.avg, metric_logger_wandb.loss_align_meter.avg)
    print('\n{}'.format(disp_str), 'val mae: {:.4f}'.format(metric_logger_wandb.refined_mae.avg))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, metric_logger_wandb


@torch.no_grad()
def inference(args, data_loader, model, device):

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()
    count = 0
    pred_ages = []
    pred_ages_coarse = []
    coarse_probs = []
    fine_probs = []

    targets = []
    img_files = []
    feats = []
    data_sources = []

    feat_1 = []
    feat_2 = []

    mae = AverageMeter()
    coarse_mae = AverageMeter()

    for image, target, _, img_dir, data_source in metric_logger.log_every(data_loader, 10, header):

        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            outputs1, outputs2, feat, feat_, _ = model(image)

        age_span = torch.arange(START_AGE, END_AGE, step=args.age_bin, dtype=torch.float32).to(image.device)
        coarse_pred = (nn.Softmax(dim=1)(outputs1) * age_span).sum(1)  # N

        age_span = torch.arange(START_AGE, END_AGE, step=1, dtype=torch.float32).to(image.device)
        final_pred = (nn.Softmax(dim=1)(outputs2) * age_span).sum(1)  # N

        mean_abs_error_coarse = torch.abs(target.squeeze() - coarse_pred.squeeze()).mean()
        mean_abs_error = torch.abs(target.squeeze() - final_pred.squeeze()).mean()

        mae.update(mean_abs_error.item(), image.size(0))
        coarse_mae.update(mean_abs_error_coarse.item(), image.size(0))

        pred_ages_coarse.extend([pred.item() + 15 for pred in coarse_pred.cpu().detach()])
        pred_ages.extend([pred.item() + 15 for pred in final_pred.cpu().detach()])

        fine_probs.append(nn.Softmax(dim=1)(outputs2).cpu().detach())
        coarse_probs.append(nn.Softmax(dim=1)(outputs1).cpu().detach())

        targets.extend([t.item() + 15 for t in target.cpu().detach()])
        feats.append(feat.cpu().detach())
        feat_1.append(feat_[0].cpu().detach())
        feat_2.append(feat_[1].cpu().detach())
        img_files.extend([t for t in img_dir])
        data_sources.extend([t for t in data_source])

        metric_logger.meters['mae_coarse'].update(mean_abs_error_coarse.item(), n=image.size(0))
        metric_logger.meters['mae_refined'].update(mean_abs_error.item(), n=image.size(0))

    metric_logger.synchronize_between_processes()

    feats = torch.concat(feats, dim=0).numpy()
    feat_1 = torch.concat(feat_1, dim=0).numpy()
    feat_2 = torch.concat(feat_2, dim=0).numpy()
    fine_probs = torch.concat(fine_probs, dim=0).numpy()
    coarse_probs = torch.concat(coarse_probs, dim=0).numpy()

    assert len(pred_ages) == len(targets)
    result = {'prediction': pred_ages, 'coarse_prediction': pred_ages_coarse, 'targets': targets, 'mean_abs_error': mae.avg, 'filenames': img_files, 'feats': feats, 
    'fine_probs': fine_probs, 'coarse_probs': coarse_probs, 'data_source': data_source, 'feat_1': feat_1, 'feat_2': feat_2}

    print('coarse mae is: {}'.format(coarse_mae.avg))
    print('mean abs error: {}'.format(mae.avg))
    return {'coarse_mae': coarse_mae.avg, 'refined_mae': mae.avg}, result


@torch.no_grad()
def inference_feat(args, data_loader, model, device):

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()
    count = 0
    
    targets = []
    img_files = []
    feats = []
    data_sources = []


    for image, target, _, img_dir, data_source in metric_logger.log_every(data_loader, 10, header):

        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            outputs1, outputs2, feat, feat_, _ = model(image)

        targets.extend([t.item() + 15 for t in target.cpu().detach()])
        feats.append(feat_[0].cpu().detach())
        img_files.extend([t for t in img_dir])
        data_sources.extend([t for t in data_source])

    metric_logger.synchronize_between_processes()

    feats = torch.concat(feats, dim=0).numpy()

    result = {'targets': targets, 'filenames': img_files, 'feats': feats}

    return result