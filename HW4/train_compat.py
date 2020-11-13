import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import argparse
import time
import copy
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils import Config
from model import SiameseNetv3
from data import get_compatibility_dataloader


def matplotlib_imshow(img, one_channel=False):
    img = np.vstack(img) # stack the pair horizontally
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def images_to_probs(net, images):
    output = net(images)
    return outputs

def plot_classes_preds(net, images, labels):
    probs=images_to_probs(net, images)
    fig=plt.figure(figsize(12,48))

    for idx in np.arange(4):
        ax=fig.add_subplot(1,4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx])
        ax.set_title(f'predict: {probs[idx]:.6f}, label: {labels[idx]}')
    return fig

def train_siamese_model(dataloaders, model, criterion, optimizer, device, num_epochs, dataset_size):
    pass
def train_model(dataloaders, model, criterion, optimizer, device, num_epochs, dataset_size, resume_epoch=0, scheduler=None):
    model.to(device)
    since=time.time()
    best_model_wts=copy.deepcopy(model.state_dict())
    best_acc=0.0
    writer=None
    if Config['tensorboard_log']:
        writer=SummaryWriter(Config['checkpoint_path'])
    for name, param in model.named_parameters():
        if name in ['fc.weight','fc.bias','classifier.weight','classifier.bias']:
            param.requires_grad=True
        else:
            param.requires_grad=False
    count=0
    for epoch in range(resume_epoch, num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-'*10)
        if epoch==int(num_epochs*0.6):
            Config['batch_size']=Config['batch_size']//2
            for name, param in model.named_parameters():
                param.requires_grad=True

            for param_group in optimizer.param_groups:
                param_group['lr']=param_group['lr']*0.7
                writer.add_scalar('learning rate', param_group['lr'], count)
        for phase in ['train','valid']:
            print(phase)
            if phase=='train':
                model.train()
            else:
                model.eval()
            running_loss=0.0
            running_corrects=0
            for inputs_one, inputs_two, labels in tqdm(dataloaders[phase]):
                # for i in range(inputs_one.shape[0]):
                #     print(labels[i].data)
                #     if labels[i].data==0:
                #         plt.imshow(inputs_one[0].permute(1, 2, 0).cpu().numpy())
                #         plt.figure()
                #         plt.imshow(inputs_one[1].permute(1, 2, 0).cpu().numpy())
                #         plt.show()
                #         return
                count+=1
                inputs_one=inputs_one.to(device)
                inputs_two=inputs_two.to(device)
                labels=labels.to(device)
                labels=labels.float()
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs=model(inputs_one, inputs_two)
                    outputs=outputs.squeeze()
                    loss=criterion(outputs, labels)
                    if phase=='train':
                        loss.backward()
                        optimizer.step()
                        if epoch > int(num_epochs*0.6):
                            scheduler.step()
                running_loss+=loss.item()*inputs_one.size(0)
                running_corrects+=torch.sum((((outputs>0.5)*1)==labels)*1)
            epoch_loss=running_loss/dataset_size[phase]
            epoch_acc=(running_corrects.double()/dataset_size[phase]).cpu().numpy()
            print(epoch_acc)
            if Config['tensorboard_log']:
                if phase=='train':
                    loss_title='train loss'
                    acc_title='train accuracy'
                elif phase=='valid':
                    loss_title='valid loss'
                    acc_title='valid accuracy'
                else:
                    loss_title='test loss'
                    acc_title='test accuracy'
                writer.add_scalar(loss_title, epoch_loss, epoch)
                writer.add_scalar(acc_title, epoch_acc, epoch)
            print(f' {phase} Loss: {epoch_loss:.4f} Acc:{epoch_acc:.4f}')
            if phase=='valid' and epoch_acc>best_acc:
                best_acc=epoch_acc
                best_model_wts=copy.deepcopy(model.state_dict())
        save_location=os.path.join(Config['checkpoint_path'],'compat_model.pth')
        torch.save(best_model_wts, save_location)
        print(f'Model saved at: {save_location}')
        time_elapsed = time.time() - since
        print(f'Time taken to complete training: {time_elapsed//60:0f}m {time_elapsed%60:0f}s')
        print(f'Best acc: {best_acc:.4f}')

def test_model(dataloaders, model, device):
    model.to(device)
    model.eval()
    running_loss=0.0
    running_corrects=0
    for inputs_one, inputs_two, labels in dataloaders['valid']:
        inputs_one = inputs_one.to(device)
        inputs_two = inputs_two.to(device)
        with torch.set_grad_enabled(False):
            outputs=model(inputs_one, inputs_two)
            outputs=outputs.squeeze()
            print(outputs, labels)
        running_corrects+=torch.sum((((outputs>0.5)*1)==labels)*1)
    epoch_acc=(running_corrects.double()/dataset_size['valid']).cpu().numpy()
    print('Test Accuracy:',epoch_acc)

def run_inference(dataloaders, model, device):
    model.to(device)
    model.eval()
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    img1=torch.unsqueeze(transform(Image.open('polyvore_outfits/images/213343990.jpg')),0)
    img2=torch.unsqueeze(transform(Image.open('polyvore_outfits/images/206270853.jpg')),0)
    plt.imshow(Image.open('polyvore_outfits/images/213343990.jpg'))
    plt.show()
    plt.imshow(Image.open('polyvore_outfits/images/206270853.jpg'))
    plt.show()
    output=model([img1,img2])
    print(output)

if __name__=='__main__':
    dataloaders, dataset_size = get_compatibility_dataloader(debug=Config['debug'], batch_size=Config['batch_size'], num_workers=Config['num_workers'])
    model=SiameseNetv3()
    # model.load_state_dict(torch.load('/home/adityan/Downloads/compat_model(1).pth',map_location=torch.device('cpu')))
    criterion=nn.BCELoss()
    optimizer=optim.Adam(model.parameters(), lr=Config['learning_rate'])
    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloaders['train']), eta_min=0)
    device=torch.device('cuda:0' if torch.cuda.is_available() and Config['use_cuda'] else 'cpu')
    # print(len(dataloaders['train']))
    if Config['resume_epoch']!=0:
        model.load_state_dict(torch.load(Config['checkpoint_file']))
    train_model(dataloaders, model, criterion, optimizer, device, num_epochs=Config['num_epochs'], dataset_size=dataset_size, resume_epoch=Config['resume_epoch'])
    # run_inference(dataloaders, model, device)
