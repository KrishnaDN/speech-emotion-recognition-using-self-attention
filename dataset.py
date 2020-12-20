#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 12:51:48 2020

@author: krishna
"""

import os
import numpy as np
import glob
import pickle
from typing import List
import re
import librosa

class Dataset:
    def __init__(self, pickle_file, save_folder, test_session: str, save_train: str, save_test: str):
        self.pickle_file = pickle_file
        self.save_folder= save_folder
        self.test_session = test_session
        self.save_train = save_train
        self.save_test = save_test
        self.emotion_id = {'hap':0,'ang':1,'sad':2,'neu':3,'exc':0}
        self.gender_id = {'M':0,'F':1}
        
        with open(self.pickle_file, 'rb') as f:
            self.data = pickle.load(f,encoding="latin1") 
    
    
    def _clean_text(self, text):
        text = text.rstrip('\r').lower()
        part_clean = re.sub("[^A-Za-z0-9']+", ' ', text).split(' ')
        all_words = []
        for item in part_clean:
            if item:
                all_words.append(item)
        join_text = ' '.join(all_words)
        return join_text
    
    
    def pre_emp(self,x):
        '''
        Apply pre-emphasis to given utterance.
        x	: list or 1 dimensional numpy.ndarray
        '''
        return np.append(x[0], np.asarray(x[1:] - 0.97 * x[:-1], dtype=np.float32))
    

    def lin_spectogram_from_wav(self,wav, hop_length, win_length, n_fft=512):
        linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
        return linear.T

    
    def _get_features(self, audio_data):
        _fs = 16000  # sampling rate
        hop_length=160
        win_length=400
        #utt, sr = librosa.load(audio_path,sr=None)
        audio_data = audio_data/np.max(audio_data)
        utt = self.pre_emp(audio_data)
        linear_spect = self.lin_spectogram_from_wav(utt, hop_length, win_length, n_fft=512)
        mag, _ = librosa.magphase(linear_spect)  # magnitude
        spec = mag.T
        logspec = np.log(spec + 1e-8).T
        return logspec
    
    
    @property
    def _get_data(self,):
        train={}
        test= {}
        for row in self.data:
            filename = row['id']
            
            sess_name = filename.split('_')[0][:-1]
            data_dict={}
            data_dict['audio_data'] = row['signal']
            data_dict['emo_label'] = self.emotion_id[row['emotion']]
            data_dict['gen_label'] = self.gender_id[filename.split('_')[0][-1]]
            data_dict['transcript'] = self._clean_text(row['transcription'])
            spec_features = self._get_features(row['signal'])
            data_dict['features'] = spec_features
            
            if sess_name == self.test_session:
                test[filename] = data_dict
            else:
                train[filename] = data_dict
        return train, test
        
    @property            
    def _save_data(self,):
        train_dict,test_dict = self._get_data
        with open(os.path.join(self.save_folder,self.save_train), 'wb') as handle:
            pickle.dump(train_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.save_folder,self.save_test), 'wb') as handle:
            pickle.dump(test_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    
if __name__=='__main__':
    pickle_file = '/media/newhd/IEMOCAP_dataset/data_collected_full.pickle'
    save_folder = '/media/newhd/IEMOCAP_dataset/IEMOCAP_processed'
    save_train = 'train_s1_s2_s4_s5.pkl'
    save_test = 'test_s3.pkl'
    test_session='Ses03'
    dataset = Dataset(pickle_file, save_folder, test_session, save_train, save_test)
    dataset._save_data
    
    
    
    
    