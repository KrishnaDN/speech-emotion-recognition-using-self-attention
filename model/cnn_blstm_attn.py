#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 12:18:02 2020

@author: krishna
"""
import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.nn import functional as F


class MaskConv(nn.Module):
    def __init__(self, seq_module):
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        for module in self.seq_module:
            x = module(x)
            mask = torch.BoolTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths):
                length = length.item()
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths


class AudioStream(nn.Module):
    def __init__(self, input_spec_size, cnn_filter_size,lstm_hidden_size=128, num_layers_lstm=2,
                 input_dropout_p=0, dropout_p=0,
                 bidirectional=True, rnn_cell='gru', variable_lengths=False):
            
        
        super(AudioStream, self).__init__()
        self.input_spec_size = input_spec_size
        self.lstm_hidden_size = lstm_hidden_size
        self.bidirectional = bidirectional
        self.num_layers_lstm = num_layers_lstm
        self.dropout_p = 0.2
        self.variable_lengths = variable_lengths
        self.num_emo_classes = 4
        self.num_gender_class = 2
        self.cnn_filter_size = cnn_filter_size
        
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))
            
    
        outputs_channel = self.cnn_filter_size
        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, outputs_channel, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(outputs_channel),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(outputs_channel, outputs_channel, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(outputs_channel),
            nn.Hardtanh(0, 20, inplace=True)
        ))
        
        rnn_input_dims = int(math.floor(input_spec_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_dims = int(math.floor(rnn_input_dims + 2 * 10 - 21) / 2 + 1)
        rnn_input_dims *= outputs_channel
        
        self.rnn =  self.rnn_cell(rnn_input_dims, self.lstm_hidden_size, self.num_layers_lstm, dropout=self.dropout_p, bidirectional=self.bidirectional)
        self.self_attn_layer = nn.TransformerEncoderLayer(d_model=self.lstm_hidden_size*2, dim_feedforward=512,nhead=8)
        self.gender_layer  = nn.Linear(self.lstm_hidden_size*4,self.num_gender_class)
        self.emotion_layer = nn.Linear(self.lstm_hidden_size*4,self.num_emo_classes)
        

    def forward(self, input_var, input_lengths=None):
        
        output_lengths = self.get_seq_lens(input_lengths)

        x = input_var # (B,1,D,T)
        x, _ = self.conv(x, output_lengths) # (B, C, D, T)
        x_size = x.size()
        x = x.view(x_size[0], x_size[1] * x_size[2], x_size[3]) # (B, C * D, T)
        x = x.transpose(1, 2).transpose(0, 1).contiguous() # (T, B, D)
        
        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths,enforce_sorted=False)
        x, h_state = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        
        x = x.transpose(0, 1) # (B, T, D)
        out = self.self_attn_layer(x)
        mu = torch.mean(out, dim=1)
        std = torch.std(out, dim=1)
        pooled = torch.cat((mu,std),dim=1)
        gen_pred = self.gender_layer(pooled)
        emo_pred = self.emotion_layer(pooled)
        return emo_pred, gen_pred
            
    

    def get_seq_lens(self, input_length):
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d :
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)

        return seq_len.int()
    
    