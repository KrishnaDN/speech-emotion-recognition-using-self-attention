#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 20:00:07 2020

@author: krishna
"""

import os
from time import time

import torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import apex.amp as amp
import argparse
from SpeechGenerator import IEMOCAPDatset, collate_fun
from models.audio_only import AudioOnly
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
import tensorboard_logger
from tensorboard_logger import log_value

global best_acc
best_acc=0


########## Argument parser
def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-train_manifest',type=str,default='/home/krishna/Krishna/Datasets/IEMOCAP/train_s1_s2_s4_s5.pkl')
    parser.add_argument('-test_manifest',type=str,default='/home/krishna/Krishna/Datasets/IEMOCAP/test_s3.pkl')
    parser.add_argument('-experiment',type=str,default='train_s1_s2_s4_s5')
    
    parser.add_argument('-input_size', action="store_true", default=257)
    parser.add_argument('-num_classes', action="store_true", default=4)
    parser.add_argument('-train_batch_size', action="store_true", default=64)
    parser.add_argument('-dev_batch_size', action="store_true", default=64)
    
    parser.add_argument('-use_gpu', action="store_true", default=True)
    parser.add_argument('-save_dir', type=str, default='save_models')
    parser.add_argument('-num_epochs', action="store_true", default=100)
    parser.add_argument('-save_interval', action="store_true", default=1000)
    parser.add_argument('-log_interval', action="store_true", default=200)
    parser.add_argument('-lr', action="store_true", default=0.001)
    args = parser.parse_args()
    return args

all_acc_nums_class_specific={}
all_acc_nums_overall={}

def train(model, device, train_loader, optimizer, loss_fun, epoch):
    model.train()  # set training mode
    gt_labels =[]
    pred_labels = []
    total_loss = []
    for batch_idx, sampled_batch in enumerate(train_loader):
        global global_step
        audio_feats = torch.stack(sampled_batch[0]).unsqueeze(1)
        
        emo_labels = torch.LongTensor(sampled_batch[2])
        gen_labels = torch.LongTensor(sampled_batch[3])
        seq_lengths = torch.LongTensor(sampled_batch[4])
        
        audio_feats, emo_labels,gen_labels, seq_lengths = audio_feats.to(device), emo_labels.to(device),gen_labels.to(device), seq_lengths.to(device)
        
        prediction_logits = model(audio_feats, seq_lengths.cpu())
        loss = loss_fun(prediction_logits,emo_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        predictions = np.argmax(prediction_logits.detach().cpu().numpy(),axis=1)
        for pred in predictions:
            pred_labels.append(pred)
        for lab in emo_labels.detach().cpu().numpy():
            gt_labels.append(lab)
        total_loss.append(loss.item())
        log_value("training loss (step-wise)", float(loss.item()),
                global_step)
        global_step = global_step +1
        
        
    log_value("training loss (epoch-wise)", np.mean(total_loss), epoch)
    acc = accuracy_score(gt_labels, pred_labels)
    print(f'Total training loss {np.mean(np.asarray(total_loss))} after {epoch}')
    print(f'Total training accuracy {acc} after {epoch}')
    


def test(model, device, test_loader, loss_fun, epoch, optimizer,exp):
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
    global best_acc
    all_preds=[]
    all_gts=[]
    with torch.no_grad():
        for batch_idx, sampled_batch in enumerate(test_loader):
            global global_step_test
            audio_feats = torch.stack(sampled_batch[0]).unsqueeze(1)
            emo_labels = torch.LongTensor(sampled_batch[2])
            gen_labels = torch.LongTensor(sampled_batch[3])
            seq_lengths = torch.LongTensor(sampled_batch[4])
            
            audio_feats, emo_labels,gen_labels, seq_lengths = audio_feats.to(device), emo_labels.to(device),gen_labels.to(device), seq_lengths.to(device)
    
            prediction_logits = model(audio_feats, seq_lengths.cpu())
            loss = loss_fun(prediction_logits,emo_labels)
            
            predictions = np.argmax(prediction_logits.detach().cpu().numpy(),axis=1)
            for pred in predictions:
                pred_labels.append(pred)
            for lab in emo_labels.detach().cpu().numpy():
                gt_labels.append(lab)
            total_loss.append(loss.item())
            
            predictions_emotion = np.argmax(prediction_logits.detach().cpu().numpy(),axis=1)
           
            for pred in predictions_emotion:
                all_preds.append(pred)
            for lab in emo_labels.detach().cpu().numpy():
                all_gts.append(lab)
                
            ########## Unweighted accuracy
            
            for k in range(len((emo_labels))):
                lab_emo = emo_labels[k]
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
            log_value("validation loss (step-wise)", float(loss.item()),
                global_step_test)
            global_step_test = global_step_test +1
            
            
            
    
    
    print(f'Total testing loss {np.mean(np.asarray(total_loss))} after {epoch}')
    acc = accuracy_score(gt_labels, pred_labels)
    print(f'Total testing accuracy {acc} after {epoch}')
    accuracy_happy=accuracy_score(happy_label,happy_pred)
    accuracy_angry=accuracy_score(angry_label,angry_pred)
    accuracy_sad=accuracy_score(sad_label,sad_pred)
    accuracy_neu=accuracy_score(neu_label ,neu_pred)
    average = np.mean([accuracy_happy,accuracy_angry,accuracy_sad,accuracy_neu])
    #print('Happy {} , Angry {}, Sad {}, Neutral {}'.format(accuracy_happy,accuracy_angry,accuracy_sad,accuracy_neu))
    print('Unweighted / class accuracy {}'.format(average))
    all_acc_nums_class_specific[epoch]=average
    accuracy_emotion = accuracy_score(all_gts,all_preds)
    print('Final Weighted test accuracy {} after {} '.format(accuracy_emotion,epoch))
    all_acc_nums_overall[epoch] = accuracy_emotion
    print('Maximum acc so far UNWEIGHTED {} -------'.format(max(all_acc_nums_class_specific.values())))
    print('Maximum acc so far WEIGHTED {} -------'.format(max(all_acc_nums_overall.values())))
    print('**************************')
    print('**************************')
    
    if average >= best_acc:
        best_acc = average
        model_save_path = os.path.join('unimodal/', exp+'_check_point_'+str(average))
        state_dict = {'model': model.state_dict(),'optimizer': optimizer.state_dict(),'epoch': epoch}
        torch.save(state_dict, model_save_path)
        
    return acc, np.mean(np.asarray(total_loss))




def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



  
def main():
    
    args = parse_args()
    setup_seed(2018)
    exp = args.experiment
    device = torch.device("cuda" if args.use_gpu else "cpu")
    print("Use", device)
    model = AudioOnly(args.input_size,num_classes=args.num_classes).to(device)
    #define loss function
    
    trainset = IEMOCAPDatset(pkl_filepath=args.train_manifest,max_len=1000)
    train_loader = DataLoader(trainset,batch_size=args.train_batch_size,num_workers=16,shuffle=True,collate_fn=collate_fun)
    
    
    testset = IEMOCAPDatset(pkl_filepath=args.test_manifest,max_len=1000)
    test_loader = DataLoader(testset,batch_size=args.dev_batch_size,num_workers=16,shuffle=False,collate_fn=collate_fun)
    
    loss_fun = loss_fun = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(),lr=args.lr, weight_decay = 5e-5)
  
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True)
    
     # Setup tensorboard logger.
    tensorboard_logger.configure(
    os.path.join('runs/exp.tb_log'))
    global global_step
    global_step = 0
    
    global global_step_test
    global_step_test = 0
    
    for epoch in range(args.num_epochs):
        
        train(model, device, train_loader, optimizer, loss_fun, epoch)
        
        test_acc, test_loss = test(model, device, test_loader, loss_fun, epoch, optimizer, exp)  # evaluate at the end of epoch
        scheduler.step(test_loss)

    
    
if __name__ == '__main__':
    main()

    
### s5 test
#Maximum acc so far UNWEIGHTED 0.5245503489857481 -------
#Maximum acc so far WEIGHTED 0.5286059629331185 -------

 ### s4 test
#Maximum acc so far UNWEIGHTED 0.5217746770984397 -------
#Maximum acc so far WEIGHTED 0.5092143549951503 -------