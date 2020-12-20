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
from model.cnn_blstm_attn import AudioStream
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.optim.lr_scheduler import ReduceLROnPlateau
########## Argument parser
parser = argparse.ArgumentParser(add_help=False)

parser.add_argument('-training_filepath',type=str,default='meta/training_s1_s2_s3_s4.txt')
parser.add_argument('-testing_filepath',type=str, default='meta/testing_s5.txt')
parser.add_argument('-input_spec_size', action="store_true", default=384)
parser.add_argument('-cnn_filter_size', action="store_true", default=64)
parser.add_argument('-num_layers_lstm', action="store_true", default=2)
parser.add_argument('-num_heads_self_attn', action="store_true", default=8)
parser.add_argument('-lstm_hidden_size', action="store_true", default=128)
parser.add_argument('-num_emo_classes', action="store_true", default=4)
parser.add_argument('-num_gender_class', action="store_true", default=2)
parser.add_argument('-batch_size', action="store_true", default=32)
parser.add_argument('-use_gpu', action="store_true", default=True)
parser.add_argument('-num_epochs', action="store_true", default=100)
parser.add_argument('-alpha', action="store_true", default=1.0)
parser.add_argument('-beta', action="store_true", default=0.25)


args = parser.parse_args()
### Data loaders
dataset_train = SpeechDataGenerator(manifest=args.training_filepath,mode='train')
dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,num_workers=16, shuffle=True,collate_fn=speech_collate) 

dataset_test = SpeechDataGenerator(manifest=args.testing_filepath,mode='test')
dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size,num_workers=16, collate_fn=speech_collate)
## Model related
if args.use_gpu:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
else:
    device='cpu'

model = AudioStream(args.input_spec_size, args.cnn_filter_size, args.lstm_hidden_size,args.num_layers_lstm)
model = model.to(device) 
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-05, betas=(0.9, 0.98), eps=1e-9)
loss = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1, verbose=True)

################################
all_acc_nums_class_specific = {}
all_acc_nums_overall = {}
for epoch in range(args.num_epochs):
    model.train()
    train_acc_list_emo = []
    train_acc_list_gen =[]
    train_loss_list=[]
    for i_batch, sample_batched in enumerate(dataloader_train):
    
        features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[0]]))
        labels_emo = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[1]])) 
        labels_gen = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[2]])) 
        lengths = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[3]])).squeeze()
        
        
        features, labels_emo, labels_gen, lengths = features.to(device,dtype=torch.float), labels_emo.to(device), labels_gen.to(device), lengths.to(device)
        features.requires_grad = True
        optimizer.zero_grad()
        preds_emo, pred_gender = model(features.unsqueeze(1), lengths.cpu()) ## only for torch 1.7 and above
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
    
    print(f'Total training accuracy {mean_acc_emo} after {epoch}')
    
    
    model.eval()  # set training mode
    gt_labels =[]
    pred_labels = []
    total_loss = []
    happy_pred = []
    happy_label=[]
    angry_pred = []
    angry_label=[]
    sad_pred = []
    sad_label=[]
    neu_pred = []
    neu_label=[]
    
    all_preds=[]
    all_gts=[]
    test_acc_list_emo=[]
    all_gts=[]
    all_preds=[]
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader_test):
            features = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[0]]))
            labels_emo = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[1]])) 
            labels_gen = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[2]])) 
            lengths = torch.from_numpy(np.asarray([torch_tensor.numpy() for torch_tensor in sample_batched[3]])).squeeze()
            
            features, labels_emo, labels_gen, lengths = features.to(device,dtype=torch.float), labels_emo.to(device), labels_gen.to(device), lengths.to(device)
            features.requires_grad = True
            optimizer.zero_grad()
            preds_emo, pred_gender = model(features.unsqueeze(1), lengths.cpu())
            emotion_loss = loss(preds_emo,labels_emo.squeeze())
            gender_loss = loss( pred_gender,labels_gen.squeeze())
            total_loss = args.alpha*emotion_loss+args.beta*gender_loss
            test_acc_list_emo.append(total_loss.item())
            
            predictions_emotion = np.argmax(preds_emo.detach().cpu().numpy(),axis=1)
            predictions_gender = np.argmax(pred_gender.detach().cpu().numpy(),axis=1)
        
            for pred in predictions_emotion:
                all_preds.append(pred)
            for lab in labels_emo.detach().cpu().numpy():
                all_gts.append(lab)
            
            
            ########## Unweighted accuracy
            for k in range(len((labels_emo))):
                lab_emo = labels_emo[k]
                pred_emo = predictions_emotion[k]
                if lab_emo==0:
                    happy_label.append(lab_emo.detach().cpu().numpy().item())
                    happy_pred.append(pred_emo)
                elif lab_emo==1:
                    angry_label.append(lab_emo.detach().cpu().numpy().item())
                    angry_pred.append(pred_emo)
                elif lab_emo==2:
                    sad_label.append(lab_emo.detach().cpu().numpy().item())
                    sad_pred.append(pred_emo)
                else:
                    neu_label.append(lab_emo.detach().cpu().numpy().item())
                    neu_pred.append(pred_emo)
            
                
    print(f'Total testing loss {np.mean(np.asarray(test_acc_list_emo))} after {epoch}')
    acc = accuracy_score(all_gts, all_preds)
    print(f'Total testing accuracy {acc} after {epoch}')
    accuracy_happy=accuracy_score(happy_label,happy_pred)
    accuracy_angry=accuracy_score(angry_label,angry_pred)
    accuracy_sad=accuracy_score(sad_label,sad_pred)
    accuracy_neu=accuracy_score(neu_label ,neu_pred)
    average = np.mean([accuracy_happy,accuracy_angry,accuracy_sad,accuracy_neu])
    #print('Happy {} , Angry {}, Sad {}, Neutral {}'.format(accuracy_happy,accuracy_angry,accuracy_sad,accuracy_neu))
    print('Unweighted / class accuracy {}'.format(average))
    all_acc_nums_class_specific[epoch]=average
    
    print('Final Weighted test accuracy {} after {} '.format(acc,epoch))
    all_acc_nums_overall[epoch] = acc
    print('Maximum acc so far UNWEIGHTED {} -------'.format(max(all_acc_nums_class_specific.values())))
    print('Maximum acc so far WEIGHTED {} -------'.format(max(all_acc_nums_overall.values())))
    print('**************************')
    print('**************************')
    test_loss= np.mean(np.asarray(test_acc_list_emo))
    scheduler.step(test_loss)


    
    print('********* Loss {} and  Emotion Accuracy {} after {} epoch '.format(mean_loss,mean_acc_emo,epoch))
    
    model_save_path = os.path.join('model_checkpoint_spec', 'check_point_'+str(epoch))
    state_dict = {'model': model.state_dict(),'optimizer': optimizer.state_dict(),'epoch': epoch}
    torch.save(state_dict, model_save_path)
    

