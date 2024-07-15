# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GCNIILayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.weight = nn.Parameter(torch.FloatTensor(in_channel, out_channel))
        self.reset_parameter()

    def reset_parameter(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, h0, alpha, beta):
        assert x.size(-1) == h0.size(-1)
        left = (1 - alpha) * torch.mm(adj, x) + alpha * h0
        right = (1 - beta) * torch.eye(self.out_channel, device=x.device) + beta * self.weight
        return left @ right


class AggAttention(nn.Module):
    def __init__(self, in_channel, att_channel):
        super().__init__()
        self.attn_proj = nn.Linear(in_channel, att_channel)
        self.act = nn.Tanh()
        self.f1 = nn.Linear(att_channel, 1)

    def forward(self, node_view_hid):
        x = self.attn_proj(node_view_hid)  # [node, view, att_hid]
        x = self.act(x)
        x = self.f1(x)  # [node, view, 1]
        w = x.mean(0).unsqueeze(0)  # [1, view, 1]
        w = F.softmax(w, 1)
        out = (node_view_hid * w).sum(1)  # [node, hid]
        return out, w


class CrossAttention(nn.Module):
    def __init__(self, S_in_channel, U_in_channel, att_channel):
        super().__init__()
        self.h0_s_linear = nn.Linear(S_in_channel, att_channel)
        self.z_linear = nn.Linear(U_in_channel, att_channel)
        self.act = nn.Tanh()
        self.f1 = nn.Linear(att_channel, 1)

    def forward(self, h0_s, z):
        # h0_s: [node, view, hid_dim]
        # z: [node, hid_dim]
        att1 = self.h0_s_linear(h0_s)  # [node, view, att]
        att2 = self.z_linear(z)  # [node, att]
        att = self.act(att1 + att2.unsqueeze(1))  # [node, view, att]
        att = self.f1(att)  # [node, view, 1]
        w = att.mean(0).unsqueeze(0)  # [1, view, 1]
        w = F.softmax(w, 1)
        out = (h0_s * w).sum(1)  # [node, view, hid_dim]
        return out, w