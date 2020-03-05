#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 18:50:31 2020

@author: krishna
"""

import os
import numpy as np
import glob

import os
import numpy as np
import glob
import os
import pickle
import codecs

#### Import pickle file

pickle_filepath = '/media/newhd/IEMOCAP_dataset/data_collected_full.pickle'
f = open(pickle_filepath, 'rb')
p_data = pickle.load(f,encoding="latin1")   
create_dict = {}
for row in p_data:
    filename = row['id']
    if row['emotion']=='exc':
        create_dict[filename] = 'hap'
    else:
        create_dict[filename] = row['emotion']
    

emotion_id = {'hap':0,'ang':1,'sad':2,'neu':3}
gender_id = {'M':0,'F':1}




##########
fid_train=open('meta/training.txt','w')
fid_test = open('meta/testing.txt','w')
data_root = '/media/newhd/IEMOCAP_dataset/raw_data/'
all_files = sorted(glob.glob(data_root+'/*.wav'))
for filepath in all_files:
    filename =  filepath.split('/')[-1]
    check_name = filename.split('_')[-1]
    if check_name=='noise.wav':
        continue
    check_sv = filename.split('_')[-1]
   
    if (check_sv[0]=='s' or check_sv[0]=='v'):
        continue
    else:
        check_filename=filepath.split('/')[-1][:-4]
        emotion = create_dict[check_filename]
        gender = check_filename.split('_')[0][-1]
        to_write = filepath+' '+str(emotion_id[emotion])+' '+str(gender_id[gender])
        check_session = check_filename.split('_')[0][:-1]
        if check_session=='Ses05':
            
            fid_test.write(to_write+'\n')
        else:
            fid_train.write(to_write+'\n')
        print(emotion,check_filename,gender)

fid_train.close()
fid_test.close()
  