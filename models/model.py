# -*- coding:utf-8 -*-
from models.layers import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASGCN(nn.Module):
    def __init__(self, n_view, n_feats, n_class,
                 n_layer, hid_dim, alpha, lamda, num_heads=8, dropout=0.5):
        super().__init__()
        self.n_view = n_view
        self.n_layer = n_layer
        self.alpha = alpha
        self.lamda = lamda
        self.num_heads = num_heads
        self.dropout = dropout

        self.proj_layers = nn.ModuleList([
            nn.Linear(n_feats[i], hid_dim) for i in range(n_view)
        ])
        self.self_att = nn.MultiheadAttention(hid_dim, num_heads=num_heads)
        self.agg_att = AggAttention(hid_dim, att_channel=hid_dim)
        self.cross_att_layers = nn.ModuleList([
            CrossAttention(hid_dim, hid_dim, att_channel=hid_dim) for _ in range(n_layer)
        ])
        self.gc_layers = nn.ModuleList([
            GCNIILayer(hid_dim, hid_dim) for _ in range(n_layer)
        ])
        self.output = nn.Linear(hid_dim, n_class)

    def forward(self, x_list_and_adj_list):
        x_list, adj_list = x_list_and_adj_list

        # proj
        h0s = []
        for i, layer in enumerate(self.proj_layers):
            h0s.append(F.dropout(F.relu(layer(x_list[i])), 0.1, training=self.training))
        h0_s = torch.stack(h0s).permute(1, 0, 2)  # [node, view, hid]
        adj0_s = torch.stack(adj_list).permute(1, 0, 2)  # [node, view, node]

        # self attention
        # [node, view, hid] [view, node, node]
        attn_o, attn_w = self.self_att(h0_s, h0_s, h0_s)
        attn_o = F.relu(attn_o)
        attn_o = F.dropout(attn_o, self.dropout, training=self.training)

        # agg attention
        # [node, hid] [1, view, 1]
        agg_z, agg_w = self.agg_att(attn_o)
        agg_z = F.relu(agg_z)
        agg_z = F.dropout(agg_z, self.dropout, training=self.training)
        agg_adj = (adj0_s * agg_w).sum(1)  # [node, node]

        # forward
        z, adj, z0 = agg_z, agg_adj, h0_s.mean(1)
        for i in range(self.n_layer):
            beta = math.log(self.lamda / (i + 1) + 1)
            z = F.relu(self.gc_layers[i](z, adj, z0, self.alpha, beta))
            z = F.dropout(z, self.dropout, training=self.training)

            # [node, hid] [1, view, 1]
            z0, agg_w = self.cross_att_layers[i](h0_s, z)
            adj = (1 - self.alpha) * adj + self.alpha * (adj0_s * agg_w).sum(1)  # [node, node]
        out = self.output(z)
        return out