#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 21:20:52 2020

@author: krishna
"""
import numpy as np
import torch
from utils import utils

class SpeechDataGenerator():
    """Speech dataset."""

    def __init__(self, manifest, mode='train'):
        """
        Read the textfile and get the paths
        """
        self.npy_links = [line.rstrip('\n').split(' ')[0] for line in open(manifest)]
        

    def __len__(self):
        return len(self.npy_links)

    def __getitem__(self, idx):
        data = np.load(self.npy_links[idx])
        audio_data = data['audio_data']
        audio_data = audio_data/np.max(audio_data)
        emo_id = data['emo_label']
        gen_id = data['gen_label']
        specgram, lens = utils.load_data(audio_data)
        sample = {'spec': torch.from_numpy(np.ascontiguousarray(specgram)), 'labels_emo': torch.from_numpy(np.ascontiguousarray(emo_id)),'labels_gen': torch.from_numpy(np.ascontiguousarray(gen_id)),
                  'lengths':torch.from_numpy(np.ascontiguousarray(lens))}
        return sample