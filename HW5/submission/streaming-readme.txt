Dataloader :
Using this dataloader can allow us to forward to this model easily. 

The model uses GRUs with input features of size 64, hidden size 10, output size 3, sequence length was varied from 20-100 and settled on 20. I had started with the RNN example notebook from the lectures to make a barebone model. The data loader has samples from all three languages in one item shuffled across for better training samples. We oversample the lower numbered languages to match English. I had used a preprocessor script to convert faulty audio files (different sampling rates, Voice Activity Detection and silence removal, stereo to mono conversion). After which I trained the model after reserving some of the audio files for testing/validation. Some files below the required 10 minute limits were removed and some had dialogues, tv shows which had to be removed to attempt to improve accuracy. I had tried different approaches to overcome silence (li Rosa trim and voice activity detector using Sox) and trimmed silence.

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
