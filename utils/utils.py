# Third Party
import librosa
import numpy as np
import torch
# ===============================================
#       code from Arsha for loading data.
# ===============================================
def load_wav(audio_filepath, sr, win_length=120000,mode='train'):
    audio_data,fs  = librosa.load(audio_filepath,sr=16000)
    if len(audio_data)<win_length:
        diff = win_length-len(audio_data)
        create_arr = np.zeros([1,diff])
        final_data  = np.concatenate((audio_data,create_arr[0]))
        audio_data = final_data
        ret_data = audio_data
    else:
        ret_data = audio_data[:win_length]
    return ret_data
    
def mel_spec_from_wav(wav, hop_length, win_length, n_mels=128):
    #linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    mel_feats=librosa.feature.melspectrogram(wav, sr=16000, n_mels=128,win_length=win_length, hop_length=hop_length)
    mel_delta = librosa.feature.delta(mel_feats)
    mel_delta2 = librosa.feature.delta(mel_delta)
    ret_spec=np.concatenate((mel_feats,mel_delta,mel_delta2))
    return ret_spec.T

def load_data(path, seg_length=120000, win_length=800, sr=16000, hop_length=400, n_fft=512, spec_len=300, mode='train'):
    wav = load_wav(path, sr=sr, mode=mode)
    linear_spect = mel_spec_from_wav(wav, hop_length, win_length, n_mels=128)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    
    spec_mag = mag_T[:, :spec_len]
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    ret_spec=(spec_mag - mu) / (std + 1e-5)
    return ret_spec


def speech_collate(batch):
    gender = []
    emotion=[]
    specs = []
    for sample in batch:
        specs.append(sample['spec'])
        emotion.append((sample['labels_emo']))
        gender.append(sample['labels_gen'])
    return specs, emotion,gender
