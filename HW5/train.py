from utils import Config
from model import LanguageDetector
from data import LanguageDatasetv2

from tqdm import tqdm
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

def train(dataloader, model, criterion, optimizer, device, scheduler, num_epochs=Config['num_epochs'], batch_size=Config['batch_size']):
    model.to(device)
    if Config['tensorboard_log']:
        writer = SummaryWriter(Config['checkpoint_path'])
    count = 0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs+1}')
        print('-'*10)

        model.train()

        running_loss=0.0
        running_corrects=0
        running_batches=0


        for inputs, labels in tqdm(dataloader):
            print(batch_size, inputs.shape, labels.shape)
            num_batches = inputs.shape[1]//batch_size
            print(num_batches)
            running_batches+=num_batches*batch_size
            for i in range(num_batches):
                inp = inputs[:,i*batch_size:(i+1)*batch_size,:][0]
                lbl = labels[:,i*batch_size:(i+1)*batch_size][0].long()
                # print(inp.shape, lbl.shape, lbl)
                inp = inp.to(device)
                lbl = lbl.to(device)
                optimizer.zero_grad()
                count += 1
                with torch.set_grad_enabled(True):
                    outputs = model(inp, device=device)
                    _, preds = torch.max(outputs[:,-1,:], dim=-1)
                    # outputs = nn.LogSoftmax(dim=1)(outputs)
                    print(outputs.shape, preds.shape,outputs[:,-1,:].squeeze(1).shape, lbl.shape) # preds, lbl)
                    loss = criterion(F.log_softmax(outputs[:,-1,:].squeeze(1),dim=1), lbl)
                    loss.backward()
                    optimizer.step()
                    running_corrects+=torch.sum(preds == lbl.data).cpu().numpy()
                    running_loss += loss.item()
        scheduler.step()

        writer.add_scalar('training_loss',running_loss/num_batches,epoch)
        writer.add_scalar('training_accuracy',running_corrects/running_batches, epoch)
        print(f'Epoch {epoch}: | Loss: {running_loss/num_batches}, Accuracy: {running_corrects/running_batches}')
        if (epoch)%3 == 0 :
            torch.save(model.state_dict(), os.path.join(Config['checkpoint_path'],'model.pth'))
    print('Finished Training')

if __name__=='__main__':
    # dataset = LanguageDatasetv2(Config['root_path'])
    # dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size=1, shuffle=True)
    # torch.save({'data_loader':dataloader}, 'dataloader.pth')
    dataloader = torch.load('dataloader.pth')['data_loader']

    model = LanguageDetector()

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    device=torch.device('cuda:0' if torch.cuda.is_available() and Config['use_cuda'] else 'cpu')
    # print(device)
    train(dataloader, model, criterion, optimizer, device, scheduler)
