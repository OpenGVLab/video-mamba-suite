# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, padding=0):
        x = x + self.pe[padding: padding + x.shape[0], :]
        return self.dropout(x)


class PositionalEncoding2D(nn.Module):

    def __init__(self, d_model, dropout=0.1, height=64, width=64):
        super(PositionalEncoding2D, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(height, width, d_model)
        position_x = torch.arange(0, width, dtype=torch.float).unsqueeze(1)
        position_y = torch.arange(0, height, dtype=torch.float).unsqueeze(1)
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, :, 0:d_model:2] = torch.sin(position_x * div_term).unsqueeze(0).repeat(height, 1, 1)
        pe[:, :, 1:d_model:2] = torch.cos(position_x * div_term).unsqueeze(0).repeat(height, 1, 1)
        pe[:, :, d_model::2] = torch.sin(position_y * div_term).unsqueeze(1).repeat(1, width, 1)
        pe[:, :, d_model + 1::2] = torch.cos(position_y * div_term).unsqueeze(1).repeat(1, width, 1)
        pe = pe.unsqueeze(0).unsqueeze(0)    #.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, padding=(0,0)):
        x = x + self.pe[:, :,
                        padding[0]: padding[0] + x.shape[2],
                        padding[1]: padding[1] + x.shape[3], :]
        return self.dropout(x)
