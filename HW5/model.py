from utils import Config
import torch
import torch.nn as nn
import torch.nn.functional as F

class LanguageDetector(nn.Module):
    def __init__(self, input_size = 64, hidden_size = Config['hidden_size'], output_size = 3):
        super(LanguageDetector, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn = nn.GRU(self.input_size, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def init_hidden(self, batch_size, device='cpu'):
        return torch.zeros(1, batch_size, self.hidden_size).to(device)

    def forward(self, x, device='cpu'):
        batch_size = x.shape[0]
        h = self.init_hidden(batch_size, device)
        out, h = self.rnn(x,h)
        out = self.fc(out)
        return out


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
