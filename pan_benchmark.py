#import sys
#import inspect

import torch
import torch.nn.functional as F

from torch.nn import Parameter
from torch_scatter import scatter_add, scatter_max, scatter_mean

from torch_geometric.utils import softmax, degree
from torch_geometric.nn import MessagePassing
from torch_geometric.data import DataLoader, Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils.num_nodes import maybe_num_nodes
#from torch_geometric.nn.pool import TopKPooling, SAGPooling

from torch.utils.data import random_split

from torch_sparse import spspmm
from torch_sparse import coalesce
from torch_sparse import eye

#from collections import OrderedDict

import os
import scipy.io as sio
import numpy as np
from optparse import OptionParser
import time
#import gdown
#import zipfile

#CUDA_visible_devices = 1

#seed = 11
#np.random.seed(seed)
#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)
#torch.cuda.manual_seed_all(seed)
##torch.cuda.seed_all(seed)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

### define convolution

class PANConv(MessagePassing):
    def __init__(self, in_channels, out_channels, filter_size=4, panconv_filter_weight=None):
        super(PANConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.m = None
        self.filter_size = filter_size
        if panconv_filter_weight is None:
            self.panconv_filter_weight = torch.nn.Parameter(0.5 * torch.ones(filter_size), requires_grad=True)

    def forward(self, x, edge_index, num_nodes=None, edge_mask_list=None):
        # x has shape [N, in_channels]
        if edge_mask_list is None:
            AFTERDROP = False
        else:
            AFTERDROP = True

        # edge_index has shape [2, E]
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        # Step 1: Path integral
        edge_index, edge_weight = self.panentropy_sparse(edge_index, num_nodes, AFTERDROP, edge_mask_list)

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)
        x_size0 = x.size(0)

        # Step 3: Compute normalization
        row, col = edge_index
        deg = degree(row, x_size0, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        norm = norm.mul(edge_weight)

        # save M
        m_list = norm.mul(edge_weight).view(-1, 1).squeeze()
        m_adj = torch.zeros(x_size0, x_size0, device=edge_index.device)
        m_adj[row, col] = m_list
        self.m = m_adj

        # Step 4-6: Start propagating messages.
        return self.propagate(edge_index, size=(x_size0, x_size0), x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out

    def panentropy(self, edge_index, num_nodes):

        # sparse to dense
        # adj = to_dense_adj(edge_index)
        adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        adj[edge_index[0, :], edge_index[1, :]] = 1

        # iteratively add weighted matrix power
        adjtmp = torch.eye(num_nodes, device=edge_index.device)
        pan_adj = self.panconv_filter_weight[0] * torch.eye(num_nodes, device=edge_index.device)

        for i in range(self.filter_size - 1):
            adjtmp = torch.mm(adjtmp, adj)
            pan_adj = pan_adj + self.panconv_filter_weight[i+1] * adjtmp

        # dense to sparse
        edge_index_new = torch.nonzero(pan_adj).t()
        edge_weight_new = pan_adj[edge_index_new[0], edge_index_new[1]]

        return edge_index_new, edge_weight_new

    def panentropy_sparse(self, edge_index, num_nodes, AFTERDROP, edge_mask_list):

        edge_value = torch.ones(edge_index.size(1), device=edge_index.device)
        edge_index, edge_value = coalesce(edge_index, edge_value, num_nodes, num_nodes)

        # iteratively add weighted matrix power
        pan_index, pan_value = eye(num_nodes, device=edge_index.device)
        indextmp = pan_index.clone().to(edge_index.device)
        valuetmp = pan_value.clone().to(edge_index.device)

        pan_value = self.panconv_filter_weight[0] * pan_value

        for i in range(self.filter_size - 1):
            if AFTERDROP:
                indextmp, valuetmp = spspmm(indextmp, valuetmp, edge_index, edge_value * edge_mask_list[i], num_nodes, num_nodes, num_nodes)
            else:
                indextmp, valuetmp = spspmm(indextmp, valuetmp, edge_index, edge_value, num_nodes, num_nodes, num_nodes)
            valuetmp = valuetmp * self.panconv_filter_weight[i+1]
            indextmp, valuetmp = coalesce(indextmp, valuetmp, num_nodes, num_nodes)
            pan_index = torch.cat((pan_index, indextmp), 1)
            pan_value = torch.cat((pan_value, valuetmp))

        return coalesce(pan_index, pan_value, num_nodes, num_nodes, op='add')


### define pooling

class PANPooling(torch.nn.Module):
    r""" General Graph pooling layer based on PAN, which can work with all layers.
    """
    def __init__(self, in_channels, ratio=0.5, pan_pool_weight=None, min_score=None, multiplier=1,
                 nonlinearity=torch.tanh, filter_size=3, panpool_filter_weight=None):
        super(PANPooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.filter_size = filter_size
        if panpool_filter_weight is None:
            self.panpool_filter_weight = torch.nn.Parameter(0.5 * torch.ones(filter_size), requires_grad=True)

        self.transform = Parameter(torch.ones(in_channels), requires_grad=True)

        if pan_pool_weight is None:
            #self.weight = torch.tensor([0.7, 0.3], device=self.transform.device)
            self.pan_pool_weight = torch.nn.Parameter(0.5 * torch.ones(2), requires_grad=True)
        else:
            self.pan_pool_weight = pan_pool_weight

    def forward(self, x, edge_index, M=None, batch=None, num_nodes=None):

        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        # Path integral
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        edge_index, edge_weight = self.panentropy_sparse(edge_index, num_nodes)

        # weighted degree
        num_nodes = x.size(0)
        degree = torch.zeros(num_nodes, device=edge_index.device)
        degree = scatter_add(edge_weight, edge_index[0], out=degree)

        # linear transform
        xtransform = torch.matmul(x, self.transform)

        # aggregate score
        x_transform_norm = xtransform #/ xtransform.norm(p=2, dim=-1)
        degree_norm = degree #/ degree.norm(p=2, dim=-1)
        score = self.pan_pool_weight[0] * x_transform_norm + self.pan_pool_weight[1] * degree_norm

        if self.min_score is None:
            score = self.nonlinearity(score)
        else:
            score = softmax(score, batch)

        perm = self.topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_weight = self.filter_adj(edge_index, edge_weight, perm, num_nodes=score.size(0))

        return x, edge_index, edge_weight, batch, perm, score[perm]

    def topk(self, x, ratio, batch, min_score=None, tol=1e-7):

        if min_score is not None:
            # Make sure that we do not drop all nodes in a graph.
            scores_max = scatter_max(x, batch)[0][batch] - tol
            scores_min = scores_max.clamp(max=min_score)

            perm = torch.nonzero(x > scores_min).view(-1)
        else:
            num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
            batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

            cum_num_nodes = torch.cat(
                [num_nodes.new_zeros(1),
                 num_nodes.cumsum(dim=0)[:-1]], dim=0)

            index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
            index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

            dense_x = x.new_full((batch_size * max_num_nodes, ), -2)
            dense_x[index] = x
            dense_x = dense_x.view(batch_size, max_num_nodes)

            _, perm = dense_x.sort(dim=-1, descending=True)

            perm = perm + cum_num_nodes.view(-1, 1)
            perm = perm.view(-1)

            k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
            mask = [
                torch.arange(k[i], dtype=torch.long, device=x.device) +
                i * max_num_nodes for i in range(batch_size)
            ]
            mask = torch.cat(mask, dim=0)

            perm = perm[mask]

        return perm

    def filter_adj(self, edge_index, edge_weight, perm, num_nodes=None):

        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        mask = perm.new_full((num_nodes, ), -1)
        i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
        mask[perm] = i

        row, col = edge_index
        row, col = mask[row], mask[col]
        mask = (row >= 0) & (col >= 0)
        row, col = row[mask], col[mask]

        if edge_weight is not None:
            edge_weight = edge_weight[mask]

        return torch.stack([row, col], dim=0), edge_weight

    def panentropy_sparse(self, edge_index, num_nodes):

        edge_value = torch.ones(edge_index.size(1), device=edge_index.device)
        edge_index, edge_value = coalesce(edge_index, edge_value, num_nodes, num_nodes)

        # iteratively add weighted matrix power
        pan_index, pan_value = eye(num_nodes, device=edge_index.device)
        indextmp = pan_index.clone().to(edge_index.device)
        valuetmp = pan_value.clone().to(edge_index.device)

        pan_value = self.panpool_filter_weight[0] * pan_value

        for i in range(self.filter_size - 1):
            #indextmp, valuetmp = coalesce(indextmp, valuetmp, num_nodes, num_nodes)
            indextmp, valuetmp = spspmm(indextmp, valuetmp, edge_index, edge_value, num_nodes, num_nodes, num_nodes)
            valuetmp = valuetmp * self.panpool_filter_weight[i+1]
            indextmp, valuetmp = coalesce(indextmp, valuetmp, num_nodes, num_nodes)
            pan_index = torch.cat((pan_index, indextmp), 1)
            pan_value = torch.cat((pan_value, valuetmp))

        return coalesce(pan_index, pan_value, num_nodes, num_nodes, op='add')


# equation 14
class PANUMPooling(torch.nn.Module):
    r""" Specific Graph pooling layer based on unnormalized M from PAN, which can only work after PANConv.
    """
    def __init__(self, in_channels, ratio=0.5, min_score=None, multiplier=1, nonlinearity=torch.tanh):
        super(PANUMPooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

    def forward(self, x, edge_index, edge_weight=None, M=None, UM=None, batch=None, num_nodes=None):

        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

        # compute score
        diag_UM = torch.diag(UM)
        score = diag_UM.squeeze()

        if self.min_score is None:
            score = self.nonlinearity(score)
        else:
            score = softmax(score, batch)

        perm = self.topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]

        edge_index, edge_weight = self.filter_adj(edge_index, edge_weight, perm, num_nodes=score.size(0))

        return x, edge_index, edge_weight, batch, perm, score[perm]

    def topk(self, x, ratio, batch, min_score=None, tol=1e-7):

        if min_score is not None:
            # Make sure that we do not drop all nodes in a graph.
            scores_max = scatter_max(x, batch)[0][batch] - tol
            scores_min = scores_max.clamp(max=min_score)

            perm = torch.nonzero(x > scores_min).view(-1)
        else:
            num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
            batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

            cum_num_nodes = torch.cat(
                [num_nodes.new_zeros(1),
                 num_nodes.cumsum(dim=0)[:-1]], dim=0)

            index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
            index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

            dense_x = x.new_full((batch_size * max_num_nodes, ), -2)
            dense_x[index] = x
            dense_x = dense_x.view(batch_size, max_num_nodes)

            _, perm = dense_x.sort(dim=-1, descending=True)

            perm = perm + cum_num_nodes.view(-1, 1)
            perm = perm.view(-1)

            k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
            mask = [
                torch.arange(k[i], dtype=torch.long, device=x.device) +
                i * max_num_nodes for i in range(batch_size)
            ]
            mask = torch.cat(mask, dim=0)

            perm = perm[mask]

        return perm

    def filter_adj(self, edge_index, edge_weight, perm, num_nodes=None):

        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        mask = perm.new_full((num_nodes, ), -1)
        i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
        mask[perm] = i

        row, col = edge_index
        row, col = mask[row], mask[col]
        mask = (row >= 0) & (col >= 0)
        row, col = row[mask], col[mask]

        if edge_weight is not None:
            edge_weight = edge_weight[mask]

        return torch.stack([row, col], dim=0), edge_weight


# equation 15
class PANXUMPooling(torch.nn.Module):
    r""" General Graph pooling layer based on PAN, which can work with all layers.
    """
    def __init__(self, in_channels, ratio=0.5, pan_pool_weight=None, min_score=None, multiplier=1,
                 nonlinearity=torch.tanh, filter_size=3, panpool_filter_weight=None):
        super(PANXUMPooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.transform = Parameter(torch.ones(in_channels), requires_grad=True)

        if pan_pool_weight is None:
            #self.weight = torch.tensor([0.7, 0.3], device=self.transform.device)
            self.pan_pool_weight = torch.nn.Parameter(0.5 * torch.ones(2), requires_grad=True)
        else:
            self.pan_pool_weight = pan_pool_weight

    def forward(self, x, edge_index, edge_weight=None, M=None, UM=None, batch=None, num_nodes=None):

        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

        # diag of unnormalized M
        diag_UM = torch.diag(UM).squeeze()

        # linear transform
        xtransform = torch.matmul(x, self.transform)

        # aggregate score
        score = self.pan_pool_weight[0] * xtransform + self.pan_pool_weight[1] * diag_UM

        if self.min_score is None:
            score = self.nonlinearity(score)
        else:
            score = softmax(score, batch)

        perm = self.topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_weight = self.filter_adj(edge_index, edge_weight, perm, num_nodes=score.size(0))

        return x, edge_index, edge_weight, batch, perm, score[perm]

    def topk(self, x, ratio, batch, min_score=None, tol=1e-7):

        if min_score is not None:
            # Make sure that we do not drop all nodes in a graph.
            scores_max = scatter_max(x, batch)[0][batch] - tol
            scores_min = scores_max.clamp(max=min_score)

            perm = torch.nonzero(x > scores_min).view(-1)
        else:
            num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
            batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

            cum_num_nodes = torch.cat(
                [num_nodes.new_zeros(1),
                 num_nodes.cumsum(dim=0)[:-1]], dim=0)

            index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
            index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

            dense_x = x.new_full((batch_size * max_num_nodes, ), -2)
            dense_x[index] = x
            dense_x = dense_x.view(batch_size, max_num_nodes)

            _, perm = dense_x.sort(dim=-1, descending=True)

            perm = perm + cum_num_nodes.view(-1, 1)
            perm = perm.view(-1)

            k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
            mask = [
                torch.arange(k[i], dtype=torch.long, device=x.device) +
                i * max_num_nodes for i in range(batch_size)
            ]
            mask = torch.cat(mask, dim=0)

            perm = perm[mask]

        return perm

    def filter_adj(self, edge_index, edge_weight, perm, num_nodes=None):

        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        mask = perm.new_full((num_nodes, ), -1)
        i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
        mask[perm] = i

        row, col = edge_index
        row, col = mask[row], mask[col]
        mask = (row >= 0) & (col >= 0)
        row, col = row[mask], col[mask]

        if edge_weight is not None:
            edge_weight = edge_weight[mask]

        return torch.stack([row, col], dim=0), edge_weight

    def panentropy_sparse(self, edge_index, num_nodes):

        edge_value = torch.ones(edge_index.size(1), device=edge_index.device)
        edge_index, edge_value = coalesce(edge_index, edge_value, num_nodes, num_nodes)

        # iteratively add weighted matrix power
        pan_index, pan_value = eye(num_nodes, device=edge_index.device)
        indextmp = pan_index.clone().to(edge_index.device)
        valuetmp = pan_value.clone().to(edge_index.device)

        pan_value = self.panpool_filter_weight[0] * pan_value

        for i in range(self.filter_size - 1):
            #indextmp, valuetmp = coalesce(indextmp, valuetmp, num_nodes, num_nodes)
            indextmp, valuetmp = spspmm(indextmp, valuetmp, edge_index, edge_value, num_nodes, num_nodes, num_nodes)
            valuetmp = valuetmp * self.panpool_filter_weight[i+1]
            indextmp, valuetmp = coalesce(indextmp, valuetmp, num_nodes, num_nodes)
            pan_index = torch.cat((pan_index, indextmp), 1)
            pan_value = torch.cat((pan_value, valuetmp))

        return coalesce(pan_index, pan_value, num_nodes, num_nodes, op='add')


# equation 16
class PANXHMPooling(torch.nn.Module):
    r""" General Graph pooling layer based on PAN, which can work with all layers.
    """
    def __init__(self, in_channels, ratio=0.5, pan_pool_weight=None, min_score=None, multiplier=1,
                 nonlinearity=torch.tanh, filter_size=3, panpool_filter_weight=None):
        super(PANXHMPooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity
        self.transform = Parameter(torch.ones(in_channels), requires_grad=True)


    def forward(self, x, edge_index, edge_weight=None, M=None, UM=None, batch=None, num_nodes=None):

        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

        # diag of unnormalized M
        diag_M = torch.diag(M).squeeze()

        # linear transform
        xtransform = torch.matmul(x, self.transform)

        # aggregate score
        score = xtransform * diag_M

        if self.min_score is None:
            score = self.nonlinearity(score)
        else:
            score = softmax(score, batch)

        perm = self.topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_weight = self.filter_adj(edge_index, edge_weight, perm, num_nodes=score.size(0))

        return x, edge_index, edge_weight, batch, perm, score[perm]

    def topk(self, x, ratio, batch, min_score=None, tol=1e-7):

        if min_score is not None:
            # Make sure that we do not drop all nodes in a graph.
            scores_max = scatter_max(x, batch)[0][batch] - tol
            scores_min = scores_max.clamp(max=min_score)

            perm = torch.nonzero(x > scores_min).view(-1)
        else:
            num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
            batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

            cum_num_nodes = torch.cat(
                [num_nodes.new_zeros(1),
                 num_nodes.cumsum(dim=0)[:-1]], dim=0)

            index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
            index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

            dense_x = x.new_full((batch_size * max_num_nodes, ), -2)
            dense_x[index] = x
            dense_x = dense_x.view(batch_size, max_num_nodes)

            _, perm = dense_x.sort(dim=-1, descending=True)

            perm = perm + cum_num_nodes.view(-1, 1)
            perm = perm.view(-1)

            k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
            mask = [
                torch.arange(k[i], dtype=torch.long, device=x.device) +
                i * max_num_nodes for i in range(batch_size)
            ]
            mask = torch.cat(mask, dim=0)

            perm = perm[mask]

        return perm

    def filter_adj(self, edge_index, edge_weight, perm, num_nodes=None):

        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        mask = perm.new_full((num_nodes, ), -1)
        i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
        mask[perm] = i

        row, col = edge_index
        row, col = mask[row], mask[col]
        mask = (row >= 0) & (col >= 0)
        row, col = row[mask], col[mask]

        if edge_weight is not None:
            edge_weight = edge_weight[mask]

        return torch.stack([row, col], dim=0), edge_weight

    def panentropy_sparse(self, edge_index, num_nodes):

        edge_value = torch.ones(edge_index.size(1), device=edge_index.device)
        edge_index, edge_value = coalesce(edge_index, edge_value, num_nodes, num_nodes)

        # iteratively add weighted matrix power
        pan_index, pan_value = eye(num_nodes, device=edge_index.device)
        indextmp = pan_index.clone().to(edge_index.device)
        valuetmp = pan_value.clone().to(edge_index.device)

        pan_value = self.panpool_filter_weight[0] * pan_value

        for i in range(self.filter_size - 1):
            #indextmp, valuetmp = coalesce(indextmp, valuetmp, num_nodes, num_nodes)
            indextmp, valuetmp = spspmm(indextmp, valuetmp, edge_index, edge_value, num_nodes, num_nodes, num_nodes)
            valuetmp = valuetmp * self.panpool_filter_weight[i+1]
            indextmp, valuetmp = coalesce(indextmp, valuetmp, num_nodes, num_nodes)
            pan_index = torch.cat((pan_index, indextmp), 1)
            pan_value = torch.cat((pan_value, valuetmp))

        return coalesce(pan_index, pan_value, num_nodes, num_nodes, op='add')



### define dropout

class PANDropout(torch.nn.Module):
    def __init__(self, filter_size=4):
        super(PANDropout, self).__init__()

        self.filter_size =filter_size

    def forward(self, edge_index, p=0.5):
        # p - probability of an element to be zeroed

        # sava all network
        #edge_mask_list = []
        edge_mask_list = torch.empty(0)
        edge_mask_list.to(edge_index.device)

        num = edge_index.size(1)
        bern = torch.distributions.bernoulli.Bernoulli(torch.tensor([p]))

        for i in range(self.filter_size - 1):
            edge_mask = bern.sample([num]).squeeze()
            #edge_mask_list.append(edge_mask)
            edge_mask_list = torch.cat([edge_mask_list, edge_mask])

        return True, edge_mask_list


### build model

class PAN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, nhid, ratio, filter_size):
        super(PAN, self).__init__()
        self.conv1 = PANConv(num_node_features, nhid, filter_size)
        self.pool1 = PANPooling(nhid, filter_size=filter_size)
##        self.drop1 = PANDropout()
        self.conv2 = PANConv(nhid, nhid, filter_size=2)
        self.pool2 = PANPooling(nhid)
##        self.drop2 = PANDropout()
        self.conv3 = PANConv(nhid, nhid, filter_size=2)
        self.pool3 = PANPooling(nhid)
        
        self.lin1 = torch.nn.Linear(nhid, nhid//2)
        self.lin2 = torch.nn.Linear(nhid//2, nhid//4)
        self.lin3 = torch.nn.Linear(nhid//4, num_classes)

        self.mlp = torch.nn.Linear(nhid, num_classes)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        perm_list = list()
        edge_mask_list = None

        x = self.conv1(x, edge_index)
        M = self.conv1.m
        x, edge_index, _, batch, perm, score_perm = self.pool1(x, edge_index, batch=batch, M=M)
        perm_list.append(perm)

#        AFTERDROP, edge_mask_list = self.drop1(edge_index, p=0.5)
        x = self.conv2(x, edge_index, edge_mask_list=edge_mask_list)
        M = self.conv2.m
        x, edge_index, _, batch, perm, score_perm = self.pool2(x, edge_index, batch=batch, M=M)
        perm_list.append(perm)
#
##        AFTERDROP, edge_mask_list = self.drop2(edge_index, p=0.5)
        x = self.conv3(x, edge_index, edge_mask_list=edge_mask_list)
        M = self.conv3.m
        x, edge_index, _, batch, perm, score_perm = self.pool3(x, edge_index, batch=batch, M=M)
        perm_list.append(perm)
        
        mean = scatter_mean(x, batch, dim=0)
        x = mean
        
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

#        x = self.mlp(x)
#        x = F.log_softmax(x, dim=-1)

        return x, perm_list

def train(model,train_loader,device):
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
        for name, param in model.named_parameters():
            # if 'pan_pool_weight' in name:
            #     param.data = param.data.clamp(0, 1)
            if 'panconv_filter_weight' in name:
                param.data = param.data.clamp(0, 1)
            if 'panpool_filter_weight' in name:
                param.data = param.data.clamp(0, 1)
    return loss_all / len(train_loader.dataset)


def test(model,loader,device):
    model.eval()

    correct = 0
    loss = 0.0
    for data in loader:
        data = data.to(device)
        out, _ = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out, data.y).item()*data.num_graphs
    return correct / len(loader.dataset), loss/len(loader.dataset)

parser = OptionParser()
parser.add_option("--dataset_name",
                  dest="dataset_name", default='PROTEINS',
                  help="the name of dataset from Pytorch Geometric, other options include PROTEINS_full, NCI1, AIDS, Mutagenicity")
parser.add_option("--phi",
                  dest="phi", default=0.3, type=np.float,
                  help="type of dataset dataset")
parser.add_option("--runs",
                  dest="runs", default=1, type=np.int,
                  help="number of runs")
parser.add_option("--batch_size", type=np.int,
                  dest="batch_size", default=32,
                  help="batch size")
parser.add_option("--L",
                  dest="L", default=4, type=np.int,
                  help="order L in MET")
parser.add_option("--learning_rate", type=np.float,
                  dest="learning_rate", default=0.005,
                  help="learning rate")
parser.add_option("--weight_decay", type=np.float,
                  dest="weight_decay", default=1e-3,
                  help="weight decay")
parser.add_option("--pool_ratio", type=np.float,
                  dest="pool_ratio", default=0.5,
                  help="proportion of nodes to be pooled")
parser.add_option("--nhid", type=np.int,
                  dest="nhid", default=64,
                  help="number of each hidden-layer neurons")
parser.add_option("--epochs", type=np.int,
                  dest="epochs", default=100,
                  help="number of epochs each run")
options, argss = parser.parse_args()
datasetname = options.dataset_name
phi = options.phi
runs = options.runs
batch_size = options.batch_size
filter_size = options.L+1
learning_rate = options.learning_rate
weight_decay = options.weight_decay
pool_ratio = options.pool_ratio
nhid = options.nhid
epochs = options.epochs

train_loss = np.zeros((runs,epochs),dtype=np.float)
val_loss = np.zeros((runs,epochs),dtype=np.float)
val_acc = np.zeros((runs,epochs),dtype=np.float)
test_acc = np.zeros(runs,dtype=np.float)
min_loss = 1e10*np.ones(runs)

# dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sv_dir = 'data/'
if not os.path.exists(sv_dir):
    os.makedirs(sv_dir)
path = os.path.join(os.path.abspath(''), 'data', datasetname)
dataset = TUDataset(path, name=datasetname)

print(len(dataset))
print(dataset.num_classes)
print(dataset.num_node_features)

num_classes = dataset.num_classes
num_node_features = dataset.num_node_features
num_edge = 0
num_node = 0
num_graph = len(dataset)

dataset1 = list()
for i in range(len(dataset)):
    data1 = Data(x=dataset[i].x, edge_index=dataset[i].edge_index, y=dataset[i].y)
    data1.num_node = dataset[i].num_nodes
    data1.num_edge = dataset[i].edge_index.size(1)
    num_node = num_node + data1.num_node
    num_edge = num_edge + data1.num_edge
    dataset1.append(data1)
dataset = dataset1

num_edge = num_edge*1.0/num_graph
num_node = num_node*1.0/num_graph

# generate training, validation and test data sets
num_training = int(num_graph*0.8)
num_val = int(num_graph*0.1)
num_test = num_graph - (num_training+num_val)

## train model
for run in range(runs):

    training_set, val_set, test_set = random_split(dataset, [num_training,num_val,num_test])
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print('***** PAN for {}, phi {} *****'.format(datasetname,phi))
    print('#training data: {}, #test data: {}'.format(num_training,num_test))
    print('Mean #nodes: {:.1f}, mean #edges: {:.1f}'.format(num_node,num_edge))
    print('Network architectur: PC-PA')
    print('filter_size: {:d}, pool_ratio: {:.2f}, learning rate: {:.2e}, weight decay: {:.2e}, nhid: {:d}'.format(filter_size,pool_ratio,learning_rate,weight_decay,nhid))
    print('batchsize: {:d}, epochs: {:d}, runs: {:d}'.format(batch_size,epochs,runs))
    print('Device: {}'.format(device))

    ## train model
    model = PAN(num_node_features,num_classes,nhid=nhid,ratio=pool_ratio,filter_size=filter_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(epochs):
        # training
        model.train()
        loss_all = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output, _ = model(data)
            loss = F.nll_loss(output, data.y)
            loss.backward()
            loss_all += data.num_graphs * loss.item()
            optimizer.step()
            for name, param in model.named_parameters():
                # if 'pan_pool_weight' in name:
                #     param.data = param.data.clamp(0, 1)
                if 'panconv_filter_weight' in name:
                    param.data = param.data.clamp(0, 1)
                if 'panpool_filter_weight' in name:
                    param.data = param.data.clamp(0, 1)
        loss = loss_all / len(train_loader.dataset)   
        train_loss[run,epoch] = loss
        # validation
        val_acc_1, val_loss_1 = test(model,val_loader,device)
        val_loss[run,epoch] = val_loss_1
        val_acc[run,epoch] = val_acc_1
        print('Run: {:02d}, Epoch: {:03d}, Val loss: {:.4f}, Val acc: {:.4f}'.format(run+1,epoch+1,val_loss[run,epoch],val_acc[run,epoch]))
        if val_loss_1 < min_loss[run]:
            # save the model and reuse later in test
            torch.save(model.state_dict(), 'latest.pth')
            min_loss[run] = val_loss_1

    # test
    model.load_state_dict(torch.load('latest.pth'))
    test_acc[run], _ = test(model,test_loader,device)
    print('==Test Acc: {:.4f}'.format(test_acc[run]))

print('==Mean Test Acc: {:.4f}'.format(np.mean(test_acc)))

t1 = time.time()
sv = datasetname + '_pcpa_runs' + str(runs) + '_phi' + str(phi) + '_time' + str(t1) + '.mat'
sio.savemat(sv,mdict={'test_acc':test_acc,'val_loss':val_loss,'val_acc':val_acc,'train_loss':train_loss,'filter_size':filter_size,'learning_rate':learning_rate,'weight_decay':weight_decay,'nhid':nhid,'batch_size':batch_size,'epochs':epochs})







