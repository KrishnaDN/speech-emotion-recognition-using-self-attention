#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 20:08:19 2020

@author: krishna
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class CNN_BLSTM_SELF_ATTN(torch.nn.Module):
    def __init__(self, input_spec_size, cnn_filter_size, num_layers_lstm, num_heads_self_attn,hidden_size_lstm, num_emo_classes,num_gender_class):
        super(CNN_BLSTM_SELF_ATTN, self).__init__()
        
        self.input_spec_size=input_spec_size
        self.cnn_filter_size=cnn_filter_size
        self.num_layers_lstm=num_layers_lstm
        self.num_heads_self_attn=num_heads_self_attn
        self.hidden_size_lstm=hidden_size_lstm
        self.num_emo_classes=num_emo_classes
        self.num_gender_class=num_gender_class
        
        self.conv_1 = nn.Conv1d(self.input_spec_size,self.cnn_filter_size,3,1)
        self.max_pooling_1 = nn.MaxPool1d(3)
        
        self.conv_2 = nn.Conv1d(self.cnn_filter_size,self.cnn_filter_size,3,1)
        self.max_pooling_2 = nn.MaxPool1d(3)
        
        ###
        self.lstm = nn.LSTM(input_size=self.cnn_filter_size, hidden_size=self.hidden_size_lstm,num_layers=self.num_layers_lstm,bidirectional=True,dropout=0.5,batch_first=True)
        ## Transformer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size_lstm*2,dim_feedforward=512,nhead=self.num_heads_self_attn)
        self.gender_layer  = nn.Linear(self.hidden_size_lstm*4,self.num_gender_class)
        self.emotion_layer = nn.Linear(self.hidden_size_lstm*4,self.num_emo_classes)


    def forward(self,inputs):
        out = self.conv_1(inputs)
        out = self.max_pooling_1(out)
        out = self.conv_2(out)
        out = self.max_pooling_2(out)
        #h_0 = Variable(torch.randn(self.num_layers_lstm,inputs.shape[0],self.hidden_size_lstm ))
        #c_0 = Variable(torch.randn(self.num_layers_lstm,inputs.shape[0],self.hidden_size_lstm ))
        out = out.permute(0, 2, 1)
        out, (final_hidden_state, final_cell_state) = self.lstm(out)
        out = self.encoder_layer(out)
        mean = torch.mean(out,1)
        std = torch.std(out,1)
        stat = torch.cat((mean,std),1)
        pred_gender=self.gender_layer(stat)
        pred_emo = self.emotion_layer(stat)
        return pred_emo,pred_gender

