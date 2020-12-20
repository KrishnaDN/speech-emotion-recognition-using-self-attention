import numpy as np
import torch
from torch.utils import data
import re
import pickle
from transformers import BertTokenizer
import librosa
import torch.nn.functional as F

tokenizer = tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
VOCAB = ('<PAD>', 'O', ',', '.', '?')
tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}




class IEMOCAPDatset():
    def __init__(self, pkl_filepath, max_len):
        with open(pkl_filepath, 'rb') as handle:
            self.data = pickle.load(handle)
        self.file_ids = list(self.data.keys())
        self.max_len = max_len
        
        
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
        
    
    
    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        datum = self.data[file_id]
        audio_data = datum['audio_data']
        emo_label = datum['emo_label']
        gen_label = datum['gen_label']
        transcript = datum['transcript']
        words = ['[CLS]']+transcript.split(' ')+['[SEP]']
        #audio_features = self._get_features(audio_data) ## Use if you don't want to use precomputed features
        audio_features = datum['features']
        text_tokens=[]
        for w in words:
            tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = tokenizer.convert_tokens_to_ids(tokens)
            text_tokens.extend(xx)
        
        if audio_features.shape[0]>self.max_len:
            tensor_feat = torch.Tensor(audio_features[:self.max_len,:])  
            seq_length = self.max_len
        else:
            tensor_feat = F.pad(torch.Tensor(audio_features), (0, 0, 0, self.max_len - audio_features.shape[0]))
            seq_length = audio_features.shape[0]
        
        return tensor_feat, text_tokens, emo_label, gen_label,seq_length
        


def collate_fun(batch):
    speech_feats = [item[0].T for item in batch]
    text_tokens = [item[1] for item in batch]
    max_length = max([len(item) for item in text_tokens])
    if max_length>=100:
        max_length=100
        
    padded_text_tokens = []
    for item in text_tokens:
        pad_text_tokens = F.pad(torch.LongTensor(item), (0,max_length - len(item)))
        padded_text_tokens.append(pad_text_tokens)
    emo_labels = [item[2] for item in batch]
    gen_labels = [item[3] for item in batch]
    seq_lengths = [item[4] for item in batch]
    
    return speech_feats, padded_text_tokens,emo_labels,gen_labels, seq_lengths
    
