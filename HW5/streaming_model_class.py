# from model import LanguageDetectorStreamer
from utils import Config

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import librosa
import sox

class LanguageDetectorStreamer(nn.Module):
    def __init__(self, input_size = 64, hidden_size = Config['hidden_size'], output_size = 3):
        super(LanguageDetectorStreamer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn = nn.GRU(self.input_size, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def init_hidden(self, batch_size, device='cuda'):
        return torch.zeros(1, batch_size, self.hidden_size).to(device)

    def forward(self, x, h, device='cuda'):
        batch_size = x.shape[0]
        out, h = self.rnn(x,h)
        out = self.fc(out)
        return out, h

if __name__=='__main__':
    audio_file = '/home/adityan/Desktop/wspace/train/train_mandarin/mandarin_0107.wav'
    class_id = 0
    sequence_length = 10

    device = torch.device('cuda:0' if torch.cuda.is_available() and Config['use_cuda'] else 'cpu')
    model = LanguageDetectorStreamer()
    model.load_state_dict(torch.load('/home/adityan/PycharmProjects/EE599_Fall2020/HW5/model.pth', map_location=device))
    model.eval()

    # print(device)
    y, sr = librosa.load(audio_file)
    y, _ = librosa.effects.trim(y, top_db=30, frame_length=256, hop_length=64)
    mat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length=int(sr*0.010)).T
    mat = np.reshape(mat[:60000], (60000//Config['sequence_length'],Config['sequence_length'],64))
    mat = torch.tensor(mat)
    print('==>',mat.shape)
    final_out=[]
    # with torch.set_grad_enabled(False):

    for i in range(mat.shape[0]//Config['batch_size']):
        # print(mat[i*Config['batch_size']:(i+1)*Config['batch_size'],:,:].shape)

        with torch.set_grad_enabled(False):
            h = model.init_hidden(Config['batch_size'],device)
            outputs, h = model(x=mat[i*Config['batch_size']:(i+1)*Config['batch_size'],:,:], h=h, device=device)
            outputs = nn.Softmax(dim=-1)(outputs)
            print(outputs[0])
            final_out.append(outputs)
        # print(h.shape)
        # print(outputs.shape)
    out = torch.cat(final_out, dim=0)
    # out = torch.mean(out, dim=1)
    # out = out[:,,:].squeeze(1)
    print('==>', out.shape)
    out = torch.reshape(out, (60000,3))
    english_probs = out[:,0].numpy()
    hindi_probs = out[:,1].numpy()
    mandarin_probs = out[:,2].numpy()

    plt.figure()
    plt.plot(hindi_probs, label='Hindi')
    plt.plot(mandarin_probs, label='Mandarin')
    plt.plot(english_probs, label='English')
    plt.legend()
    plt.title('mandarin_0107.wav')
    plt.xlabel('Timesteps')
    plt.ylabel('Probability (Softmax Activation)')
    plt.show()

    torch.save(model, 'streaming-model.pth')
    print(english_probs.shape)
