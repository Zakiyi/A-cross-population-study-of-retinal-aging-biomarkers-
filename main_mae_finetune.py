# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import yaml
import shutil
import argparse
import datetime
import json
import numpy as np
import os
import time
import wandb
import torch.nn as nn
from pathlib import Path
import pandas as pd
import sys
sys.path.append(os.getcwd())
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import models.util.lr_decay as lrd
import models.util.misc as misc
from models.util.datasets import build_dataset
from models.util.pos_embed import interpolate_pos_embed
from models.util.misc import NativeScalerWithGradNormCount as NativeScaler
from models.utils.ranking_loss import batchwise_ranking_regularizer
from data_proc.snapshot_dataset import Retinal_Dataset

from models.mae_vit_retinal import RegModel

from trains.engine_finetune import train_one_epoch, evaluate, inference

import umap
import umap.plot as up
import matplotlib.pyplot as plt

START_AGE = 0
END_AGE = 77

def plot_umap(args, targets, feats, feat_1, feat_2):
    targets = np.array(targets)
    map = umap.UMAP(n_neighbors=120, min_dist=0.2).fit(feats)
    up.points(map, labels=targets//5, theme='fire')
    plt.savefig(args.output_dir + '/umap.png', dpi=600)

    map = umap.UMAP(n_neighbors=120, min_dist=0.2).fit(feat_1)
    up.points(map, labels=targets//5, theme='fire')
    plt.savefig(args.output_dir + '/umap1.png', dpi=600)

    map = umap.UMAP(n_neighbors=120, min_dist=0.2).fit(feat_2)
    up.points(map, labels=targets//5, theme='fire')
    plt.savefig(args.output_dir + '/umap2.png', dpi=600)

def ordinal_loss(embeddings, labels, margin=0.5, batchwise_ranking=False):
    if batchwise_ranking:
        loss = batchwise_ranking_regularizer(embeddings, labels, lambda_val=2)
    else:
        batch_size = embeddings.shape[0]
        label_dis = torch.abs(labels.view(-1, 1).repeat(1, batch_size) - labels.view(1, -1).repeat(batch_size, 1))
        label_dis_shift = torch.roll(label_dis, 1, dims=1)

        distance = torch.sqrt(torch.sum(torch.square(embeddings.unsqueeze(0) - embeddings.unsqueeze(1)), dim=-1) + 1e-12)
        distance_shift = torch.roll(distance, 1, dims=1)
        # print('distance ', distance.data.max(), distance.data.min())

        label_gap = label_dis - label_dis_shift
        distance_gap = distance_shift - distance
        # print('label gap ', label_gap.abs().data.max(), label_gap.abs().data.min())
        # print('distance gap ', distance_gap.data.max(), distance_gap.data.min())

        margin = label_gap.abs().detach() * margin
        # multiply sign of label_gap.abs is because the two samples can come from same class
        #print('distance_gap ', distance_gap.abs().min(), distance_gap.abs().max())
        loss_margin = torch.nn.functional.relu(torch.sign(label_gap).float() * distance_gap + margin)
        loss_mask = (loss_margin > 0).to(dtype=loss_margin.dtype)

        loss = loss_margin * torch.sign(label_gap.abs()).float()
        # print('loss ', loss.sum().item())

        if torch.sum(torch.sign(label_gap.abs()).float() * loss_mask) > 0:

            loss = loss.sum() / torch.sum(torch.sign(label_gap.abs()).float() * loss_mask)
        else:
            loss = torch.tensor(0.).to(embeddings.device)

    return loss


class MeanVarianceLoss(nn.Module):
    def __init__(self, lambda_1, lambda_2, start_age, end_age, age_bin):
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.start_age = start_age
        self.end_age = end_age
        self.age_bin = age_bin

    def forward(self, input, target):
        target = target.type(torch.FloatTensor).to(input.device)
        p = nn.Softmax(dim=1)(input)

        # mean loss
        a = torch.arange(self.start_age, self.end_age, self.age_bin, dtype=torch.float32).to(input.device)
        mean = torch.squeeze((p * a).sum(1, keepdim=True), dim=1)
        mse = (mean - target) ** 2

        mean_loss = mse.mean() / 2.0

        # variance loss
        b = (a[None, :] - mean[:, None]) ** 2
        variance_loss = (p * b).sum(1, keepdim=True).mean()

        return self.lambda_1 * mean_loss + self.lambda_2 * variance_loss


def alignment_loss(feats, targets):
    distance = torch.sqrt(torch.sum(torch.square(feats.unsqueeze(0) - feats.unsqueeze(1)), dim=-1) + 1e-12)
    class_eq = targets.unsqueeze(1) - targets.unsqueeze(0)  # batch * batch

    pos_dis = torch.where(class_eq == 0, distance, torch.zeros_like(distance)).sum()
    loss = 0.005 * pos_dis

    return loss


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=45, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)

    parser.add_argument('--data-dir', default='/home/zyi/retinal_age/snap_dataset', help='data directory')
    parser.add_argument('--csv-file', default='data/mixed_data_final_png.csv', help='path to csv file')

    parser.add_argument('--age-bin', type=int, default=10, help='age bin size')
    parser.add_argument('--backbone', default='vit-base-imagenet', help='core cnn model')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--lambda1', type=float, default=0.2, help='coefficient for mean loss')
    parser.add_argument('--lambda2', type=float, default=0.05, help='coefficient for var loss')
    parser.add_argument('--embedding-dim', type=int, default=768, help='coefficient for var loss')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',  help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,  help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.2, metavar='PCT', help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--test-mode', default=False, action='store_true', help='resume training')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',  help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,  help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.65, help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N', help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT', help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME', help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0., help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0, help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0, help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None)
    parser.add_argument('--mixup_prob', type=float, default=1.0, help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5)
    parser.add_argument('--mixup_mode', type=str, default='batch', help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--debug', default=False, action='store_true', help='turn on debug mode')
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool', help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str, help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int, help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False, help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',  help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    """step 1: setup data"""
    df = pd.read_csv(args.csv_file)
    if args.debug:
        print(df.head())

    df_train = df[df.is_train == 1]
    df_valid = df[df.is_train == 1]

    df_test = df[df.is_train == 0]

    dataset_test = Retinal_Dataset(df_test, size=args.input_size, is_train=False, test_mode=True, debug=args.debug)

    dataset_train = Retinal_Dataset(df_train, size=args.input_size, is_train=True, test_mode=False, debug=args.debug)
    dataset_val = Retinal_Dataset(df_valid, size=args.input_size, is_train=False, test_mode=False, debug=args.debug)

    print('Train size: {}, valid size: {}'.format(len(dataset_train), len(dataset_val)))

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        print("Sampler_train = %s" % str(sampler_train))

        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        
        if args.dist_eval:
            if len(dataset_test) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_test = torch.utils.data.DistributedSampler(dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
                
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None

    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax, prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, 
            mode=args.mixup_mode, label_smoothing=args.smoothing, num_classes=args.nb_classes)

    """step 2: setup model & loss_fn & optimizer"""
    num_class = len(torch.arange(START_AGE, END_AGE, step=args.age_bin))  # (END_AGE - START_AGE) // AGE_BIN_SIZE
    model = RegModel(args.backbone, num_class=num_class, img_size=args.input_size, age_bin=args.age_bin, embedding_dim=args.embedding_dim, dropout=args.drop_path)

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load mae pretrained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
    else:
        print('load ssl pretrained model!!!')
        checkpoint = timm.create_model(args.backbone, pretrained=True)
        checkpoint_model = checkpoint.state_dict()

    interpolate_pos_embed(model, checkpoint_model) 

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    if args.smoothing > 0.:
        ce_loss = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        ce_loss = torch.nn.CrossEntropyLoss()

    criterion1 = MeanVarianceLoss(args.lambda1, args.lambda2, START_AGE, END_AGE, args.age_bin)
    criterion3 = MeanVarianceLoss(args.lambda1, args.lambda2, START_AGE, END_AGE, 1)
    criterions = {'loss_coarse': criterion1, 'ce_loss': ce_loss, 'loss_refined': criterion3, 'align_loss': alignment_loss, 'ordinal_loss': ordinal_loss}

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                        no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                        layer_decay=args.layer_decay
                                        )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    runs_name = f'mae-pretrained-{args.backbone}-cascade-reg-bz-{eff_batch_size}-align'
    args.output_dir = 'runs/baselines/regression_align/mae-pretrained'

    if not args.debug and not args.eval:
        wandb.init(name=runs_name,
                   project="Retinal_age_prediction",
                   notes="baselines",
                   tags=["snap_dataset"],
                   config=args
                   )

    if not args.debug:
        if os.path.exists(args.resume):
            args.output_dir = os.path.dirname(args.resume)
        else:
            args.output_dir = os.path.join(args.output_dir, runs_name)
            os.makedirs(args.output_dir, exist_ok=True)

        file = open(os.path.join(args.output_dir, "config_file.yml"), "w")
        yaml.dump(args.__dict__, file)
        shutil.copy(os.path.realpath(__file__), args.output_dir)
        shutil.copy('models/mae_vit_retinal.py', args.output_dir)
    else:
        args.output_dir = 'dropmae/runs/debug'
        os.makedirs(args.output_dir, exist_ok=True)

    if args.eval:
        test_stats, result = inference(args, data_loader_test, model, device)
        torch.save(result, os.path.join(args.output_dir, 'result.pt'))
        #plot_umap(args, result['targets'], result['feats'], result['feat_1'], result['feat_2'])
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    min_mae = 100.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats, train_metrics = train_one_epoch(model, criterions, data_loader_train, optimizer, device, epoch,
                                                     loss_scaler, args.clip_grad, mixup_fn, log_writer=log_writer, args=args)

        val_stats, val_metrics = evaluate(args, data_loader_val, model, criterions, device, epoch)

        # if args.output_dir and epoch == args.epochs:
        #     misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        if val_metrics.refined_mae.avg < min_mae:
            min_mae = val_metrics.refined_mae.avg
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)
        
        if log_writer is not None:
            log_writer.add_scalar('perf/val coarse_mae', val_stats['mae_coarse'], epoch)
            log_writer.add_scalar('perf/val refined_mae', val_stats['mae_refined'], epoch)
            log_writer.add_scalar('perf/val loss', val_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'val_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if not args.debug:
            wandb.log({"val_loss": val_metrics.overall_loss.avg,
                       "val_loss1": val_metrics.loss_stage1_meter.avg,
                       "val_loss2": val_metrics.loss_stage2_meter.avg,
                       "val_loss_ordinal": val_metrics.loss_ordinal_meter.avg,
                       "val_loss_proxy": val_metrics.loss_align_meter.avg,
                       'val_mae': val_metrics.refined_mae.avg,
                       'val_mae_coarse': val_metrics.coarse_mae.avg,

                       "train_loss": train_metrics.overall_loss.avg,
                       "train_loss1": train_metrics.loss_stage1_meter.avg,
                       "train_loss2": train_metrics.loss_stage2_meter.avg,
                       "train_loss_ordinal": train_metrics.loss_ordinal_meter.avg,
                       "train_loss_proxy": train_metrics.loss_align_meter.avg,
                       'train_mae': train_metrics.coarse_mae.avg,
                       'train_mae_coarse': train_metrics.refined_mae.avg
                       })

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
