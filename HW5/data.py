import torch
from torch.utils.data import Dataset

import librosa
import os
import numpy as np
from tqdm import tqdm
import random

class LanguageDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path
        self.processed_audio_files=[]

        audio_files = os.listdir(os.path.join(self.root_path,'train_english'))
        audio_files = [f for f in audio_files if f.endswith('.wav')]
        self.english_features = []
        for file in tqdm(audio_files):
            y, sr = librosa.load(os.path.join(self.root_path, 'train_english', file), sr=16000)

            if sr!=16000:
                continue
            else:
                mat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length=int(sr*0.010)).T
                self.english_features.append(mat)
        self.english_features = np.concatenate(self.english_features)

        audio_files = os.listdir(os.path.join(self.root_path,'train_mandarin'))
        audio_files = [f for f in audio_files if f.endswith('.wav')]
        self.mandarin_features = []
        for file in tqdm(audio_files):
            y, sr = librosa.load(os.path.join(self.root_path, 'train_mandarin', file), sr=16000)
            if sr!=16000:
                continue
            else:
                mat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length=int(sr*0.010)).T
                self.mandarin_features.append(mat.T)
        self.mandarin_features = np.stack(self.mandarin_features)
        audio_files = os.listdir(os.path.join(self.root_path,'train_hindi'))
        audio_files = [f for f in audio_files if f.endswith('.wav')]
        self.hindi_features = []
        for file in tqdm(audio_files):
            y, sr = librosa.load(os.path.join(self.root_path, 'train_hindi', file), sr=16000)
            if sr!=16000:
                continue
            else:
                mat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length=int(sr*0.010)).T
                self.hindi_features.append(mat.T)
        self.hindi_features = np.stack(self.hindi_features)
        print(f'Shapes: English:{self.english_features.shape}, Mandarin:{self.mandarin_features.shape}, Hindi:{self.hindi_features.shape}')

    def __len__(self):
        return len(self.processed_audio_files)

    def __getitem__(self, idx):
        y, sr = librosa.load(self.processed_audio_files[idx], sr=16000)
        mat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length=int(sr*0.010))
        print(y.shape, sr, mat,shape)

class LanguageDatasetv2(Dataset):
    def __init__(self,root_path, sequence_length=6):
        self.root_path = root_path
        self.sequence_length = sequence_length
        self.processed_audio_files = {
            'english':[],
            'hindi':[],
            'mandarin':[]
        }

        audio_files = os.listdir(os.path.join(self.root_path,'train_english'))
        audio_files = [f for f in audio_files if f.endswith('.wav')]
        for file in tqdm(audio_files):
            y, sr = librosa.load(os.path.join(self.root_path, 'train_english', file), sr=16000)
            if sr!=16000:
                continue
            else:
                y, _ = librosa.effects.trim(y, top_db=30, frame_length=256, hop_length=64)
                if(librosa.feature.mfcc(y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025),hop_length=int(sr*0.010)).shape[1]<60000):
                    continue
                self.processed_audio_files['english'].append(os.path.join(self.root_path, 'train_english', file))

        audio_files = os.listdir(os.path.join(self.root_path,'train_mandarin'))
        audio_files = [f for f in audio_files if f.endswith('.wav')]
        for file in tqdm(audio_files):
            y, sr = librosa.load(os.path.join(self.root_path, 'train_mandarin', file), sr=16000)
            if sr!=16000:
                continue
            else:
                y, _ = librosa.effects.trim(y, top_db=30, frame_length=256, hop_length=64)
                if(librosa.feature.mfcc(y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025),hop_length=int(sr*0.010)).shape[1]<60000):
                    continue
                self.processed_audio_files['mandarin'].append(os.path.join(self.root_path, 'train_mandarin', file))

        audio_files = os.listdir(os.path.join(self.root_path,'train_hindi'))
        audio_files = [f for f in audio_files if f.endswith('.wav')]
        for file in tqdm(audio_files):
            y, sr = librosa.load(os.path.join(self.root_path, 'train_hindi', file), sr=16000)
            if sr!=16000:
                continue
            else:
                y, _ = librosa.effects.trim(y, top_db=30, frame_length=256, hop_length=64)
                if(librosa.feature.mfcc(y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025),hop_length=int(sr*0.010)).shape[1]<60000):
                    continue
                self.processed_audio_files['hindi'].append(os.path.join(self.root_path, 'train_hindi', file))
        # print(len(self.processed_audio_files['english']),
        #     len(self.processed_audio_files['hindi']),
        #     len(self.processed_audio_files['mandarin']))
    def __len__(self):
        return max(len(self.processed_audio_files['english']),
            len(self.processed_audio_files['hindi']),
            len(self.processed_audio_files['mandarin'])
            )
    def __getitem__(self, idx):
        if len(self.processed_audio_files['english'])<=idx:
            english = random.choice(self.processed_audio_files['english'])
        else:
            english = self.processed_audio_files['english'][idx]

        if len(self.processed_audio_files['mandarin'])<=idx:
            mandarin = random.choice(self.processed_audio_files['mandarin'])
        else:
            mandarin = self.processed_audio_files['mandarin'][idx]

        if len(self.processed_audio_files['hindi'])<=idx:
            hindi = random.choice(self.processed_audio_files['hindi'])
        else:
            hindi = self.processed_audio_files['hindi'][idx]

        y_eng, sr = librosa.load(english, sr=16000, mono=True)
        y_eng, _ = librosa.effects.trim(y_eng, top_db=30, frame_length=256, hop_length=64)
        mat_eng = librosa.feature.mfcc(y=y_eng, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length=int(sr*0.010)).T
        mat_eng = np.reshape(mat_eng[:60000], (60000//self.sequence_length,self.sequence_length,64))

        y_man, sr = librosa.load(mandarin, sr=16000, mono=True)
        y_man, _ = librosa.effects.trim(y_man, top_db=40, frame_length=256, hop_length=64)
        mat_man = librosa.feature.mfcc(y=y_man, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length=int(sr*0.010)).T
        mat_man = np.reshape(mat_man[:60000], (60000//self.sequence_length,self.sequence_length,64))

        y_hin, sr = librosa.load(hindi, sr=16000, mono=True)
        y_hin,_ = librosa.effects.trim(y_hin, top_db=40, frame_length=256, hop_length=64)
        mat_hin = librosa.feature.mfcc(y=y_hin, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length=int(sr*0.010)).T
        mat_hin = np.reshape(mat_hin[:60000], (60000//self.sequence_length,self.sequence_length,64))

        X = np.concatenate([mat_eng, mat_man, mat_hin])

        lbl_eng = np.zeros((mat_eng.shape[0],))
        # lbl_eng[:,:,0]=1
        lbl_hin = np.ones((mat_hin.shape[0], ))
        # lbl_hin[:,:,1]=1
        lbl_man = np.ones((mat_man.shape[0], ))*2
        # lbl_man[:,:,2]=1
        Y = np.concatenate([lbl_eng, lbl_man, lbl_hin])
        p = np.random.permutation(X.shape[0])

        return X[p], Y[p]

class LanguageDatasetv3(Dataset):
    def __init__(self, root_path, sequence_length=8):
        self.root_path = root_path
        self.sequence_length = sequence_length
        self.processed_audio_files = {
            'english':[],
            'hindi':[],
            'mandarin':[]
        }
        audio_files = os.listdir(os.path.join(self.root_path,'train_english'))
        audio_files = [f for f in audio_files if f.endswith('.wav')]
        for file in tqdm(audio_files):
            print(file)
            y, sr = librosa.load(os.path.join(self.root_path, 'train_english', file), sr=16000)
            if sr!=16000:
                continue
            else:
                y, _ = librosa.effects.trim(y, top_db=30, frame_length=256, hop_length=64)
                mat = librosa.feature.mfcc(y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025),hop_length=int(sr*0.010))
                if mat.shape[1]>=60000:
                    mat = mat.T
                    mat = np.reshape(mat[:60000], (60000//self.sequence_length,self.sequence_length,64))
                    self.processed_audio_files['english'].append(mat)
        self.processed_audio_files['english'] = np.concatenate(self.processed_audio_files['english'])
        audio_files = os.listdir(os.path.join(self.root_path,'train_mandarin'))
        audio_files = [f for f in audio_files if f.endswith('.wav')]
        for file in tqdm(audio_files):
            y, sr = librosa.load(os.path.join(self.root_path, 'train_mandarin', file), sr=16000)
            if sr!=16000:
                continue
            else:
                y, _ = librosa.effects.trim(y, top_db=30, frame_length=256, hop_length=64)
                mat = librosa.feature.mfcc(y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025),hop_length=int(sr*0.010))
                if mat.shape[1]>=60000:
                    mat = mat.T
                    mat = np.reshape(mat[:60000], (60000//self.sequence_length,self.sequence_length,64))
                    self.processed_audio_files['mandarin'].append(mat)
        self.processed_audio_files['mandarin'] = np.concatenate(self.processed_audio_files['mandarin'])
        audio_files = os.listdir(os.path.join(self.root_path,'train_hindi'))
        audio_files = [f for f in audio_files if f.endswith('.wav')]
        for file in tqdm(audio_files):
            y, sr = librosa.load(os.path.join(self.root_path, 'train_hindi', file), sr=16000)
            if sr!=16000:
                continue
            else:
                y, _ = librosa.effects.trim(y, top_db=30, frame_length=256, hop_length=64)
                mat = librosa.feature.mfcc(y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025),hop_length=int(sr*0.010))
                if mat.shape[1]>=60000:
                    mat = mat.T
                    mat = np.reshape(mat[:60000], (60000//self.sequence_length,self.sequence_length,64))
                    self.processed_audio_files['hindi'].append(mat)
        self.processed_audio_files['hindi'] = np.concatenate(self.processed_audio_files['hindi'])
        self.english_len = self.processed_audio_files['english'].shape[0]
        self.mandarin_len = self.processed_audio_files['mandarin'].shape[0]
        self.hindi_len = self.processed_audio_files['hindi'].shape[0]
        self.processed_audio_files = np.concatenate(self.processed_audio_files['english'],
                                                self.processed_audio_files['hindi'],
                                                self.processed_audio_files['mandarin'])
        print(self.processed_audio_files.shape)
    def __len__(self):
        return self.processed_audio_files.shape[0]
    def __getitem__(self, idx):
        return self.processed_audio_files[idx], 0
