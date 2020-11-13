import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
import os
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
import random

from utils import Config

class polyvore_dataset:
    def __init__(self):
        self.root_dir = Config['root_path']
        self.image_dir = os.path.join(self.root_dir,'images')
        self.transforms = self.get_data_transforms()

    def get_data_transforms(self):
        data_transforms={
            'train': transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
        }
        return data_transforms

    def create_dataset(self):
        meta_file=open(os.path.join(self.root_dir, Config['meta_file']),'r')
        meta_json=json.load(meta_file)
        id_to_category={}
        for k,v in tqdm(meta_json.items()):
            id_to_category[k] = v['category_id']

        files=os.listdir(self.image_dir)
        X=[] ; y=[]
        for x in files:
            if x[:-4] in id_to_category:
                X.append(x)
                y.append(int(id_to_category[x[:-4]]))
        label_encoder=LabelEncoder()
        y=label_encoder.fit_transform(y)
        print(f'len of X:{len(X)}, # of categories: {max(y)+1}')
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)
        for i in X_test:
            print(i)
        return X_train, X_test, y_train, y_test, max(y)+1, label_encoder, id_to_category

class polyvore_compatibility_dataset:
    def __init__(self):
        self.root_dir=Config['root_path']
        self.image_dir=os.path.join(self.root_dir,'images')
        self.transforms = self.get_data_transforms()

    def get_data_transforms(self):
        data_transforms={
            'train': transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
        }
        return data_transforms
    def gather_data(self, comp_file, out_map_file):
        X=[];y=[];
        data={'positives':[],'negatives':[]}
        if os.path.exists(os.path.join(self.root_dir,comp_file)) and os.path.exists(os.path.join(self.root_dir,out_map_file)):
            outfit_map=json.load(open(os.path.join(self.root_dir,out_map_file),'r'))
            processed_outfit_map={} #for easy fetching, whose idea was it to even make it a list linear search instead of hash fetch :/
            for item in outfit_map:
                processed_outfit_map[item['set_id']]=[it['item_id'] for it in item['items']]
            del outfit_map
            compatible_outfits_data_train=open(os.path.join(self.root_dir,comp_file),'r').readlines()
            for line in compatible_outfits_data_train:
                line=line.strip().split(' ')
                sample=line[0]
                items=[el.strip().split('_') for el in line[1:] if el!='']
                # print(processed_outfit_map)
                items=[processed_outfit_map[el[0]][int(el[1])-1] for el in items]
                pairs=list(combinations(items,2))
                if sample==1:
                    data['positives']+=pairs
                else:
                    data['negatives']+=pairs
        return data

    def create_dataset(self):
        X_train=[]; X_valid=[]; X_test=[];
        X_train = self.gather_data('compatibility_train.txt','train.json')
        X_valid = self.gather_data('compatibility_valid.txt','valid.json')
        X_test = self.gather_data('compatibility_test_hw.txt','test.json')
        return X_train, X_valid, X_test

class PolyvoreCompatibilityDataset(Dataset):
    def __init__(self, X, transform, randomize_order=True):
        self.X=X['positives'][::2]+X['negatives'][1::2]
        y_pos=[1]*len(X['positives'])
        y_neg=[0]*len(X['negatives'])
        self.y=y_pos[::2]+y_neg[1::2]
        self.transform = transform
        self.randomize_order = randomize_order # probably possibly maybe might affect speed
        self.image_dir = os.path.join(Config['root_path'],'images')
    def __len__(self):
        return len(self.X)
    def __getitem__(self, item):
        file_path_one=os.path.join(self.image_dir, self.X[item][0]+'.jpg')
        file_path_two=os.path.join(self.image_dir, self.X[item][1]+'.jpg')
        ims=[self.transform(Image.open(file_path_one)), self.transform(Image.open(file_path_two))]
        if self.randomize_order:
            random.shuffle(ims)
        return ims[0], ims[1], self.y[item]

class PolyvoreDataset(Dataset):
    def __init__(self, X, y, transform):
        self.X=X
        self.y=y
        self.transform=transform
        self.image_dir=os.path.join(Config['root_path'],'images')
    def __len__(self):
        return len(self.X)
    def __getitem__(self, item):
        file_path=os.path.join(self.image_dir, self.X[item])
        return self.transform(Image.open(file_path)), self.y[item]
def get_compatibility_dataloader(debug, batch_size, num_workers):
    dataset=polyvore_compatibility_dataset()
    transforms=dataset.get_data_transforms()
    X_train, X_valid, X_test = dataset.create_dataset()
    if debug:
        X_train={'positives':X_train['positives'][:60],'negatives':X_train['negatives'][:60]}
        X_valid={'positives':X_valid['positives'][:60],'negatives':X_valid['negatives'][:60]}
        X_test={'positives':X_test['positives'][:60],'negatives':X_test['negatives'][:60]}
        train_set=PolyvoreCompatibilityDataset(X_train, transform=transforms['train'])
        valid_set=PolyvoreCompatibilityDataset(X_valid, transform=transforms['train'])
        test_set=PolyvoreCompatibilityDataset(X_test, transform=transforms['test'])
        dataset_size = {'train': 60, 'valid': 60, 'test': 60}

    else:
        train_set=PolyvoreCompatibilityDataset(X_train, transform=transforms['train'])
        valid_set=PolyvoreCompatibilityDataset(X_valid, transform=transforms['train'])
        test_set=PolyvoreCompatibilityDataset(X_test, transform=transforms['test'])
        dataset_size = {'train': len(X_train), 'valid':len(X_valid),'test': len(X_test)}
    datasets={'train':train_set, 'valid':valid_set,'test':test_set}
    dataloaders={x: DataLoader(datasets[x],
                    shuffle=True if x=='train' or x=='valid' else False,
                    batch_size=batch_size,
                    num_workers=num_workers) for x in ['train', 'valid', 'test']}
    return dataloaders, dataset_size

def get_dataloader(debug, batch_size, num_workers):
    dataset=polyvore_dataset()
    transforms=dataset.get_data_transforms()
    X_train, X_test, y_train, y_test, classes, label_enc, id_to_category = dataset.create_dataset()
    if debug==True:
        train_set=PolyvoreDataset(X_train[:100], y_train[:100], transform=transforms['train'])
        test_set=PolyvoreDataset(X_test[:100], y_test[:100], transform=transforms['test'])
        dataset_size = {'train': 100, 'test': 100}
    else:
        train_set=PolyvoreDataset(X_train, y_train, transform=transforms['train'])
        test_set=PolyvoreDataset(X_test, y_test, transform=transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}
    datasets={'train': train_set, 'test':test_set}
    dataloaders={x:DataLoader(datasets[x],
                            shuffle=True if x=='train' else False,
                            batch_size=batch_size,
                            num_workers=num_workers)
                            for x in ['train','test']}
    return dataloaders, classes, dataset_size, label_enc, id_to_category
