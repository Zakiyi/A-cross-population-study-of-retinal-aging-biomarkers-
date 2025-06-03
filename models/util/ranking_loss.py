# Copyright (c) 2021-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#######################################################################################################################
# Code is based on the Blackbox Combinatorial Solvers (https://github.com/martius-lab/blackbox-backprop) implementation
# from https://github.com/martius-lab/blackbox-backprop by Marin Vlastelica et al.
#######################################################################################################################
import torch
import numpy as np
import random
import torch.nn.functional as F
from memory_profiler import profile


def stable_argsort(arr, dim=-1, descending=False):
    """
    More details about the stable version of pytorch argsort implementation, 
    please refer to https://github.com/pytorch/pytorch/issues/38681
    """
    arr_np = arr.detach().cpu().numpy()
    if descending:
        indices = np.argsort(-arr_np, axis=dim, kind='stable')
    else:
        indices = np.argsort(arr_np, axis=dim, kind='stable')
    return torch.from_numpy(indices).long().to(arr.device)

def flipp(T, dim):
    inv_idx = torch.arange(T.size(dim)-1, -1, -1).long().to(T.device)
    # or equivalently 
    # inv_idx = torch.range(tensor.size(0)-1, 0, -1).long()
    inv_tensor = T.index_select(dim, inv_idx)
    return inv_tensor
    # or equivalently
    # inv_tensor = T[inv_idx]
    # return inv_tensor
    
def rank(seq):
    return stable_argsort(flipp(stable_argsort(seq), 1))

def rank_normalised(seq):
    return (rank(seq) + 1).float() / seq.size()[1]

class TrueRanker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sequence, lambda_val):
        rank = rank_normalised(sequence)
        ctx.lambda_val = lambda_val
        ctx.save_for_backward(sequence, rank)
        return rank

    @staticmethod
    def backward(ctx, grad_output):
        sequence, rank = ctx.saved_tensors
        assert grad_output.shape == rank.shape
        sequence_prime = sequence + ctx.lambda_val * grad_output
        rank_prime = rank_normalised(sequence_prime)
        gradient = -(rank - rank_prime) / (ctx.lambda_val + 1e-8)
        return gradient, None


def batchwise_ranking_regularizer(features, targets, lambda_val=1, similarity='cosine'):
    loss = 0

    # Reduce ties and boost relative representation of infrequent labels by computing the 
    # regularizer over a subset of the batch in which each label appears at most once
    batch_unique_targets = torch.unique(targets)

    if len(batch_unique_targets) < len(targets):
        sampled_indices = []
        for target in batch_unique_targets:
            sampled_indices.append(random.choice((targets == target).nonzero()[:,0]).item())
        x = features[sampled_indices]
        y = targets[sampled_indices]
    else:
        x = features
        y = targets

    # Compute feature similarities
    if similarity == 'cosine':
        xxt = torch.matmul(F.normalize(x.view(x.size(0),-1)), F.normalize(x.view(x.size(0),-1)).permute(1,0))
    else:
        xxt = -torch.sqrt(torch.sum(torch.square(x.unsqueeze(0) - x.unsqueeze(1)), dim=-1) + 1e-12)
        
    # Compute ranking loss
    for i in range(len(y)):
        label_ranks = rank_normalised(-torch.abs(y[i] - y).unsqueeze(1).transpose(0,1))
        feature_ranks = TrueRanker.apply(xxt[i].unsqueeze(dim=0), lambda_val)
        loss += F.mse_loss(feature_ranks, label_ranks)
    
    return loss