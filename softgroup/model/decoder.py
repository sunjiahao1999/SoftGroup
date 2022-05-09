# Copyright (c) Gorilla-Lab. All rights reserved.
import functools
from turtle import forward
from typing import Dict
from numpy import indices

import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_min, scatter_mean, scatter_max

import pointgroup_ops
import gorilla
import gorilla.nn as gn
import gorilla3d.nn as g3n
from spconv.pytorch import SparseConvTensor
from spconv.pytorch.modules import SparseModule
import math


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class CrossAttentionLayer(SparseModule):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        dropout=0.0,
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, source, query, batch_offsets, attn_masks=None):
        '''
        source (B*N, d_model)
        batch_offsets List[int] (b+1)
        query Tensor (b, 100, d_model)
        '''
        B = len(batch_offsets) - 1
        outputs = []
        for i in range(B):
            start_id = batch_offsets[i]
            end_id = batch_offsets[i + 1]
            k = v = source[start_id:end_id].unsqueeze(0)  # (1, n, d_model)
            if attn_masks:
                output, _ = self.attn(query[i].unsqueeze(0), k, v, attn_mask = attn_masks[i])  # (1, 100, d_model)
            else:
                output, _ = self.attn(query[i].unsqueeze(0), k, v)
            self.dropout(output)
            output = output + query[i]
            self.norm(output)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0)  # (b, 100, d_model)
        return outputs


class SelfAttentionLayer(SparseModule):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        dropout=0.0,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        x Tensor (b, 100, c)
        '''
        output, _ = self.attn(x, x, x)
        output = self.dropout(output) + x
        output = self.norm(output)
        return output


class FFN(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.0, activation_fn='relu'):
        super().__init__()
        if activation_fn == 'relu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        elif activation_fn == 'gelu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        output = self.net(x)
        output = output + x
        output = self.norm(output)
        return output


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.channels = channels

        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, features, indices, spatial_shape):
        assert features.shape[1] == self.channels, 'channels must be same'
        # normalize
        indices = indices / (spatial_shape + 1e-6) * 2 * math.pi

        sin_inp_x = torch.einsum('i,j->ij', indices[:, 0], self.inv_freq)
        sin_inp_y = torch.einsum('i,j->ij', indices[:, 1], self.inv_freq)
        sin_inp_z = torch.einsum('i,j->ij', indices[:, 2], self.inv_freq)

        embedding = torch.zeros_like(features, device=features.device)
        embedding[:, 0::2] = sin_inp_x.sin() + sin_inp_y.sin() + sin_inp_z.sin()
        embedding[:, 1::2] = sin_inp_x.cos() + sin_inp_y.cos() + sin_inp_z.cos()
        return embedding


class SuperPointDecoder(nn.Module):
    '''
    in_channels List[int] (4,) [64,96,128,160]
    '''

    def __init__(
        self,
        num_layer=6,
        num_query=100,
        num_class=18,
        in_channel=32,
        d_model=256,
        nhead=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='relu',
        iter_pred=False,
        attn_mask=False,
    ):
        super().__init__()
        self.num_layer = num_layer
        self.input_proj = nn.Sequential(nn.Linear(in_channel, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.query = nn.Embedding(num_query, d_model)
        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for i in range(num_layer):
            self.cross_attn_layers.append(CrossAttentionLayer(d_model, nhead, dropout))
            self.self_attn_layers.append(SelfAttentionLayer(d_model, nhead, dropout))
            self.ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))
        self.out_norm = nn.LayerNorm(d_model)
        self.out_cls = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, num_class + 1))
        self.x_mask = nn.Sequential(nn.Linear(in_channel, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask

    def get_mask(self, query, mask_feats, batch_offsets):
        pred_masks = []
        attn_masks = []
        for i in range(len(batch_offsets) - 1):
            start_id, end_id = batch_offsets[i], batch_offsets[i + 1]
            mask_feat = mask_feats[start_id:end_id]
            pred_mask = torch.einsum('nd,md->nm', query[i], mask_feat)
            if self.attn_mask:
                # attn_mask = torch.ones_like(pred_mask)
                # _, idx = pred_mask.topk(100, dim=-1)
                # attn_mask.scatter_(1, idx, 0)
                # attn_mask = attn_mask.bool()
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)

        return pred_masks, attn_masks

    def prediction_head(self, query, mask_feats, batch_offsets):
        query = self.out_norm(query)
        pred_labels = self.out_cls(query)
        pred_masks, attn_masks = self.get_mask(query, mask_feats, batch_offsets)
        return pred_labels, pred_masks, attn_masks

    def forward_iter_pred(self, x, batch_offsets):
        '''
        x [B*M, inchannel]
        '''
        prediction_labels = []
        prediction_masks = []
        inst_feats = self.input_proj(x)
        mask_feats = self.x_mask(x)
        B = len(batch_offsets) - 1
        query = self.query.weight.unsqueeze(0).repeat(B, 1, 1)  # (b, n, d_model)
        pred_labels, pred_masks, attn_masks = self.prediction_head(query, mask_feats, batch_offsets)
        prediction_labels.append(pred_labels)
        prediction_masks.append(pred_masks)
        for i in range(self.num_layer):
            query = self.cross_attn_layers[i](inst_feats, query, batch_offsets, attn_masks)
            query = self.self_attn_layers[i](query)
            query = self.ffn_layers[i](query)
            pred_labels, pred_masks, attn_masks = self.prediction_head(query, mask_feats, batch_offsets)
            prediction_labels.append(pred_labels)
            prediction_masks.append(pred_masks)
        return {
            'labels': pred_labels,
            'masks': pred_masks,
            'aux_outputs': [{"labels": a, "masks": b} for a, b in zip(prediction_labels[:-1], prediction_masks[:-1])],
        }

    def forward_simple(self, x, batch_offsets):
        inst_feats = self.input_proj(x)
        mask_feats = self.x_mask(x)
        B = len(batch_offsets) - 1
        query = self.query.weight.unsqueeze(0).repeat(B, 1, 1)  # (b, n, d_model)
        for i in range(self.num_layer):
            query = self.cross_attn_layers[i](inst_feats, query, batch_offsets)
            query = self.self_attn_layers[i](query)
            query = self.ffn_layers[i](query)
        query = self.out_norm(query)
        pred_labels = self.out_cls(query)
        pred_masks, _ = self.get_mask(query, mask_feats, batch_offsets)
        return {'labels': pred_labels, 'masks': pred_masks}

    def forward(self, x, batch_offsets):
        if self.iter_pred:
            return self.forward_iter_pred(x, batch_offsets)
        else:
            return self.forward_simple(x, batch_offsets)


if __name__ == '__main__':
    scale = torch.arange(100)
    meshgrid3d = torch.stack(torch.meshgrid(scale, scale, scale, indexing='ij'), dim=-1)
    meshgrid3d = meshgrid3d.reshape(-1, 3)
    sampling_coords = torch.randperm(meshgrid3d.shape[0])[:20000]
    coords = meshgrid3d[sampling_coords]
    batch_idx = []
    for i in range(4):
        batch = torch.zeros(5000, 1) + i
        batch_idx.append(batch)
    batch_idx = torch.cat(batch_idx, dim=0)
    coords = torch.cat((batch_idx, coords), dim=1).int().cuda()
    inputs = []
    for i in range(5):
        features = torch.randn(20000, 32 * (5 - i)).cuda()
        input = spconv.SparseConvTensor(features, coords, spatial_shape=[100, 100, 100], batch_size=4)
        inputs.append(input)
    batch_offsets = [0, 5000, 10000, 15000, 20000]
    # net = PositionalEncoding3D(channels=32).cuda()
    # net(
    #     inputs[-1].features,
    #     inputs[-1].indices[:, 1:],
    #     torch.tensor(inputs[-1].spatial_shape, device=inputs[-1].indices.device),
    # )

    # model = VMF()
    # model.cuda()
    # output = model(inputs, input_map, batch_offsets)
    # pass
