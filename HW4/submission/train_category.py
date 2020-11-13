import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torchvision import transforms
import torchvision

import argparse
import time
import copy
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils import Config
from model import SusNetv2
from data import get_dataloader

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels, label_enc, category_to_id):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    print(id_to_category)
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx])
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            category_to_id[str(label_enc.inverse_transform([preds[idx]])[0])],
            probs[idx] * 100.0,
            category_to_id[str(label_enc.inverse_transform([labels[idx]])[0])]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig
def train_model(dataloaders, model, criterion, optimizer, device, label_enc, id_to_category, scheduler, num_epochs, dataset_size, resume_epoch=0):
    category_to_id={v:k for k,v in id_to_category.items()}
    model.to(device)
    since=time.time()
    best_model_wts=copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss=[]
    test_loss=[]
    train_acc=[]
    test_acc=[]
    writer=None
    draw_model=-1
    if Config['tensorboard_log']:
        writer=SummaryWriter(Config['checkpoint_path'])
        draw_model=0
    count=0
    for epoch in range(resume_epoch, num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-'*10)

        for phase in ['train', 'test']:
            print(phase)
            if phase=='train':
                model.train()
            else:
                model.eval()
            running_loss=0.0
            running_corrects=0
            for inputs,labels in tqdm(dataloaders[phase]):
                inputs=inputs.to(device)
                labels=labels.to(device)
                if draw_model==0:
                    model.eval()
                    writer.add_graph(model,inputs)
                    model.train()
                    draw_model+=1
                optimizer.zero_grad()
                count+=1
                with torch.set_grad_enabled(phase=='train'):
                    outputs=model(inputs)
                    _, pred=torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if Config['use_cuda']:
                        l2_regularization = torch.tensor(0.).cuda()
                    else:
                        l2_regularization = torch.tensor(0.)

                    for param in model.parameters():
                        l2_regularization+=torch.norm(param,2)**2
                    loss += 1e-5*l2_regularization
                    if phase=='train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        for param_group in optimizer.param_groups:
                            writer.add_scalar('learning rate', param_group['lr'], count)
                running_loss+=loss.item()*inputs.size(0)
                running_corrects+=torch.sum(pred==labels.data)
            epoch_loss=running_loss/dataset_size[phase]
            epoch_acc=running_corrects.double()/dataset_size[phase]
            if phase=='train':
              train_loss.append(epoch_loss)
              train_acc.append(epoch_acc)
              if Config['tensorboard_log']:
                  writer.add_scalar('training loss', epoch_loss, epoch)
                  writer.add_scalar('training accuracy', epoch_acc, epoch)
                  for param_group in optimizer.param_groups:
                      writer.add_scalar('learning rate', param_group['lr'], count)
            else:
              test_loss.append(epoch_loss)
              test_loss.append(epoch_acc)
              if Config['tensorboard_log']:
                  writer.add_scalar('test loss', epoch_loss, epoch)
                  writer.add_scalar('test accuracy', epoch_acc, epoch)
            print(f' {phase} Loss: {epoch_loss:.4f} Acc:{epoch_acc:.4f}')
            if phase=='test' and epoch_acc>best_acc:
                best_acc=epoch_acc
                best_model_wts=copy.deepcopy(model.state_dict())

        save_location=os.path.join(Config['checkpoint_path'],'model.pth')
        torch.save({
          'model':model,
          'label':label_enc.classes_,
          'epoch':epoch
        }, save_location)
        print(f'Model saved at: {save_location}')
    model.load_state_dict(best_model_wts)
    save_location=os.path.join(Config['checkpoint_path'],'model.pth')
    torch.save({
      'model':model,
      'label':label_enc.classes_,
      'epoch':epoch
    }, save_location)
    print(f'Model saved at: {save_location}')
    time_elapsed = time.time() - since
    print(f'Time taken to complete training: {time_elapsed//60:0f}m {time_elapsed%60:0f}s')
    print(f'Best acc: {best_acc:.4f}')
    return best_model_wts, train_loss, test_loss, train_acc, test_acc


if __name__=='__main__':
    dataloaders, classes, dataset_size, label_enc, id_to_category = get_dataloader(debug=Config['debug'], batch_size=Config['batch_size'], num_workers=Config['num_workers'])

    device=torch.device('cuda:0' if torch.cuda.is_available() and Config['use_cuda'] else 'cpu')
    if Config['use_custom_model']:
        #model=SusNetv2(device, classes)
        model=torchvision.models.mobilenet_v2(pretrained=False)
        model.classifier[1]=nn.Linear(1280,classes)
    else:
        from model import model
        num_ftrs=model.fc.in_features
        model.fc=nn.Linear(num_ftrs, classes)
        if Config['finetune']:
            for name, param in model.named_parameters():
                if name in ['fc.bias','fc.weight']:
                    param.requires_grad=True
                else:
                    param.requires_grad=False

    if Config['resume_epoch']!=0:
        model.load_state_dict(torch.load(Config['checkpoint_file'],map_location=device)['model'])

    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(), lr=Config['learning_rate'])
    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloaders['train']), eta_min=Config['learning_rate'])
    best_model_wts, train_loss, test_loss, train_acc, test_acc = train_model(dataloaders, model, criterion, optimizer, device,label_enc, id_to_category, scheduler, num_epochs=Config['num_epochs'],dataset_size=dataset_size)
    # generate_category_txt(dataloaders, dataset_size, model, label_enc, id_to_category, device)
