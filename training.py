#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 21:13:37 2020

@author: krishna
"""


import torch
import numpy as np


from torch.utils.data import DataLoader   
from SpeechDataGenerator_spec import SpeechDataGenerator
import torch.nn as nn
import os
import argparse
from utils.utils import speech_collate
import numpy as np
from torch import optim
from model.attn_cnn_blstm import CNN_BLSTM_SELF_ATTN
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')



########## Argument parser
parser = argparse.ArgumentParser(add_help=False)

parser.add_argument('-training_filepath',type=str,default='meta/training.txt')
parser.add_argument('-testing_filepath',type=str, default='meta/testing.txt')
parser.add_argument('-input_spec_size', action="store_true", default=384)
parser.add_argument('-cnn_filter_size', action="store_true", default=64)
parser.add_argument('-num_layers_lstm', action="store_true", default=2)
parser.add_argument('-num_heads_self_attn', action="store_true", default=8)
parser.add_argument('-hidden_size_lstm', action="store_true", default=60)
parser.add_argument('-num_emo_classes', action="store_true", default=4)
parser.add_argument('-num_gender_class', action="store_true", default=2)
parser.add_argument('-batch_size', action="store_true", default=100)
parser.add_argument('-use_gpu', action="store_true", default=True)
parser.add_argument('-num_epochs', action="store_true", default=100)
parser.add_argument('-alpha', action="store_true", default=1.0)
parser.add_argument('-beta', action="store_true", default=1.0)


args = parser.parse_args()

### Data loaders
dataset_train = SpeechDataGenerator(manifest=args.training_filepath,mode='train')
dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,collate_fn=speech_collate) 

dataset_test = SpeechDataGenerator(manifest=args.testing_filepath,mode='test')
dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size,collate_fn=speech_collate)
## Model related
if args.use_gpu:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
else:
    device='cpu'

model = CNN_BLSTM_SELF_ATTN(args.input_spec_size, args.cnn_filter_size, args.num_layers_lstm, args.num_heads_self_attn,args.hidden_size_lstm, args.num_emo_classes,args.num_gender_class)
model = model.to(device) 
#model.load_state_dict(torch.load('model_checkpoints/check_point_old')['model'])
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-05, betas=(0.9, 0.98), eps=1e-9)
loss = nn.CrossEntropyLoss()


################################
for epoch in range(args.num_epochs):
    model.train()
    train_acc_list_emo = []
    train_acc_list_gen =[]
    train_loss_list=[]
    for i_batch, sample_batched in enumerate(dataloader_train):
        #print(sample_batched)
        
        features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[0]]))
        labels_emo = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[1]])) 
        labels_gen = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[2]])) 
        
        features, labels_emo, labels_gen = features.to(device,dtype=torch.float), labels_emo.to(device), labels_gen.to(device)
        features.requires_grad = True
        optimizer.zero_grad()
        preds_emo, pred_gender = model(features)
        emotion_loss = loss(preds_emo,labels_emo.squeeze())
        gender_loss = loss( pred_gender,labels_gen.squeeze())
        total_loss = args.alpha*emotion_loss+args.beta*gender_loss
        total_loss.backward()
        optimizer.step()
        
        train_loss_list.append(total_loss.item())
        predictions_emotion = np.argmax(preds_emo.detach().cpu().numpy(),axis=1)
        predictions_gender = np.argmax(pred_gender.detach().cpu().numpy(),axis=1)
        
        #print(total_loss.item())
        #predictions= preds.detach().cpu().numpy()>=0.5
        accuracy_emotion = accuracy_score(labels_emo.detach().cpu().numpy(),predictions_emotion)
        accuracy_gender = accuracy_score(labels_gen.detach().cpu().numpy(),predictions_gender)
        
        train_acc_list_emo.append(accuracy_emotion)
        train_acc_list_gen.append(accuracy_gender)
        if i_batch%20==0:
            print('Loss {} after {} iteration'.format(np.mean(np.asarray(train_loss_list)),i_batch))
        
        
    mean_loss = np.mean(np.asarray(train_loss_list))
    mean_acc_emo = np.mean(np.asarray(train_acc_list_emo))
    
    print('********* Loss {} and  Emotion Accuracy {} after {} epoch '.format(mean_loss,mean_acc_emo,epoch))
    
    model_save_path = os.path.join('model_checkpoint_spec', 'check_point_'+str(epoch))
    state_dict = {'model': model.state_dict(),'optimizer': optimizer.state_dict(),'epoch': epoch}
    torch.save(state_dict, model_save_path)
    
    
    model.eval()
    cum_acc=0.0
    test_acc_list_emo=[]
    all_gts=[]
    all_preds=[]
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader_test):
            features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[0]]))
            labels_emo = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[1]])) 
            labels_gen = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[2]])) 
            features, labels_emo, labels_gen = features.to(device,dtype=torch.float), labels_emo.to(device), labels_gen.to(device)
            preds_emo, pred_gender = model(features)
            #total_loss = loss(preds, labels.squeeze())
            predictions_emotion = np.argmax(preds_emo.detach().cpu().numpy(),axis=1)
            predictions_gender = np.argmax(pred_gender.detach().cpu().numpy(),axis=1)
        
            for pred in predictions_emotion:
                all_preds.append(pred)
            for lab in labels_emo.detach().cpu().numpy():
                all_gts.append(lab)
        #print(total_loss.item())
        #predictions= preds.detach().cpu().numpy()>=0.5
            #accuracy_emotion = accuracy_score(labels_emo.detach().cpu().numpy(),predictions_emotion)
            #accuracy_gender = accuracy_score(labels_gen.detach().cpu().numpy(),predictions_gender)
            #prediction = np.argmax(preds.detach().cpu().numpy(),axis=1)
            #test_acc_list_emo.append(accuracy_emotion)
        #mean_test_acc = np.mean(np.asarray(test_acc_list_emo)) 
        accuracy_emotion = accuracy_score(all_gts,all_preds)
        print('********* Final test accuracy {} after {} '.format(accuracy_emotion,epoch))
        


