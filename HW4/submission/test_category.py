import torch
from torchvision import transforms
import torch.nn.functional as F

from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils import Config
from data import get_dataloader

def generate_category_txt(dataloaders, dataset_size, model, label_enc, id_to_category, device, save_path='category.txt'):
    result=open(save_path,'w')
    model=model.to(device)
    model.eval()
    test_set = open(os.path.join(Config['root_path'],'test_category_hw.txt'),'r').readlines()
    test_set = [file.strip() for file in test_set]
    print(len(test_set),dataset_size['test'])
    transform = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
            ])
    with torch.set_grad_enabled(False):
        running_corrects=0
        for file in tqdm(test_set):
            img = Image.open(os.path.join(Config['root_path'],'images',file.strip()+'.jpg'))
            img = transform(img)
            img = img.to(device)
            img = img.unsqueeze(0)
            label = id_to_category[file.strip()]
            outputs = model(img)
            # print(outputs.shape)
            _, pred = torch.max(F.softmax(outputs, 1),1)

            if int(label_enc[pred.cpu().numpy()[0]])==int(label):
              running_corrects+= 1
            # print(f'{file}, {pred.cpu().numpy()[0]}, {label}\n')
            result.write(f'{file}, {label_enc[pred.cpu().numpy()[0]]}, {label}\n')
    epoch_acc=running_corrects/len(test_set)
    print(f'Testing completed, accuracy is {epoch_acc}')
    # torch.save({
    #   'model':model.to('cpu'),
    #   'label':label_enc.classes_
    # }, Config['checkpoint_file'])
    result.close()

if __name__=='__main__':
    dataloaders, classes, dataset_size, label_enc, id_to_category = get_dataloader(debug=Config['debug'], batch_size=Config['batch_size'], num_workers=Config['num_workers'])
    device=torch.device('cuda:0' if torch.cuda.is_available() and Config['use_cuda'] else 'cpu')
    saved_model=torch.load(Config['checkpoint_file'], map_location=device)
    model = saved_model['model']
    generate_category_txt(dataloaders, dataset_size, model, saved_model['label'], id_to_category, device)
