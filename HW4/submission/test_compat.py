import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

import os
import json
import numpy as np
from PIL import Image

from model import SiameseNetv3
from utils import Config

if __name__=='__main__':
    model=SiameseNetv3()
    model.load_state_dict(Config['checkpoint_file'])
    test_set = open(os.path.join(Config['root_path'],'compatibility_test_hw.txt'),'r').readlines()
    test_set =[file.strip() for file in test_set if file!='']
    comp_file = 'compatibility_test_hw.txt'
    out_map_file = os.path.join(Config['root_path'],'test.json')
    outfit_map=json.load(open(os.path.join(Config['root_path'],out_map_file),'r'))
    processed_outfit_map={} #for easy fetching, whose idea was it to even make it a list linear search instead of hash fetch :/
    for item in outfit_map:
        processed_outfit_map[item['set_id']]=[it['item_id'] for it in item['items']]
    del outfit_map
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    output_file = open('pair_compatibility.txt','w')
    for line in test_set:
        line=line.strip().split(' ')
        items=[el.strip().split('_') for el in line if el!='']
        # print(processed_outfit_map)
        items=[processed_outfit_map[el[0]][int(el[1])-1] for el in items]
        pairs=list(combinations(items,2))
        scores=[]
        for pair in pairs:
            img1=Image.open(Config['root_path'],'images',pair[0]+'.jpg')
            img2=Image.open(Config['root_path'],'images',pair[1]+'.jpg')
            out = model(transform(img1).unsqueeze(0), transform(img2).unsqueeze(0))
            if out.data > 0.5:
                output_file.write(pair[0]+' '+pair[1]+' 1\n')
            else:
                output_file.write(pair[0]+' '+pair[1]+' 0\n')
            scores.append(out.data)
        print('Outfit score:'+str(np.mean(scores))+'\n')
    output_file.close()
