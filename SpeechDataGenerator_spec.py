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
        self.audio_links = [line.rstrip('\n').split(' ')[0] for line in open(manifest)]
        self.labels_emotion = [int(line.rstrip('\n').split(' ')[1]) for line in open(manifest)]
        self.labels_gender = [int(line.rstrip('\n').split(' ')[2]) for line in open(manifest)]
        

    def __len__(self):
        return len(self.audio_links)

    def __getitem__(self, idx):
        audio_link =self.audio_links[idx]
        emo_id = self.labels_emotion[idx]
        gen_id = self.labels_gender[idx]
        specgram = utils.load_data(audio_link)
        sample = {'spec': torch.from_numpy(np.ascontiguousarray(specgram)), 'labels_emo': torch.from_numpy(np.ascontiguousarray(emo_id)),'labels_gen': torch.from_numpy(np.ascontiguousarray(gen_id))}
        return sample