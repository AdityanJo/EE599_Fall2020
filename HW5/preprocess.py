import librosa
import os
import numpy as np
from tqdm import tqdm
import sox

from utils import Config

if __name__=='__main__':
    audio_files = os.listdir(os.path.join(Config['root_path'], 'train_english'))
    audio_files = [f for f in audio_files if f.endswith('.wav')]
    for file in tqdm(audio_files):
        # y, sr = librosa.load(os.path.join(Config['root_path'],'train_english', file), sr=16000, mono=True)
        tfm = sox.Transformer()
        tfm.vad(initial_pad=0.3)
        tfm.set_output_format(rate=16000, channels=1)
        tfm.build(input_filepath=os.path.join(Config['root_path'],'train_english', file),
            output_filepath=os.path.join(Config['root_path'],'out_train_english', file))

    audio_files = os.listdir(os.path.join(Config['root_path'], 'train_hindi'))
    audio_files = [f for f in audio_files if f.endswith('.wav')]
    for file in tqdm(audio_files):
        # y, sr = librosa.load(os.path.join(Config['root_path'],'train_english', file), sr=16000, mono=True)
        tfm = sox.Transformer()
        tfm.vad(initial_pad=0.3)
        tfm.set_output_format(rate=16000, channels=1)
        tfm.build(input_filepath=os.path.join(Config['root_path'],'train_hindi', file),
            output_filepath=os.path.join(Config['root_path'],'out_train_hindi', file))

    audio_files = os.listdir(os.path.join(Config['root_path'], 'train_mandarin'))
    audio_files = [f for f in audio_files if f.endswith('.wav')]
    for file in tqdm(audio_files):
        # y, sr = librosa.load(os.path.join(Config['root_path'],'train_english', file), sr=16000, mono=True)
        tfm = sox.Transformer()
        tfm.vad(initial_pad=0.3)
        tfm.set_output_format(rate=16000, channels=1)
        tfm.build(input_filepath=os.path.join(Config['root_path'],'train_mandarin', file),
            output_filepath=os.path.join(Config['root_path'],'out_train_mandarin', file))
