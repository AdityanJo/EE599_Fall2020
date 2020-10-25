import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.linear1=nn.Linear(20,20)
        self.linear2=nn.Linear(20,1)

    def forward(self, x):
        x=self.linear1(x)
        x=F.relu(x)
        x=self.linear2(x)
        return F.sigmoid(x)


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.hidden_dim = 16
        self.embedding=nn.Embedding(3,self.hidden_dim)
        self.fc = nn.Linear(self.hidden_dim, 1)
        self.fc_out = nn.Linear(20,1)

    def forward(self,x):
        embedded=self.embedding(x)
        out=self.fc(embedded)
        out=self.fc_out(out.squeeze())
        print('out',out.shape)
        return out

if __name__=='__main__':
    mlp=MLP()
    # mlp=RNN()
    data_file='binary_random_20fa.hdf5'
    with h5py.File(data_file,'r') as hf:
        human = hf['human'][:]
        machine = hf['machine'][:]
    x_data = np.vstack([human, machine])
    human_y=np.ones(5100).T
    machine_y=np.zeros(5100).T
    y_data=np.concatenate([human_y,machine_y])
    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    criterion=nn.BCELoss()
    optimizer=torch.optim.Adam(mlp.parameters(),lr=1e-3)

    running_loss=0.0
    count=0
    train_acc=[]
    valid_acc=[]
    for epoch in range(200):
        for i in range(int(x_train.shape[0]/16)):
            optimizer.zero_grad()
            x_batch=torch.Tensor(x_train[i:i+16])
            y_batch=torch.Tensor(y_train[i:i+16])

            y_pred=mlp(x_batch)

            loss=criterion(y_pred,y_batch)
            # l2_regularization=torch.tensor(0.)
            # l1_regularization=torch.tensor(0.)
            # for param in mlp.parameters():
                # l1_regularization+=torch.norm(param,1)
                # l2_regularization+=torch.norm(param)
            # loss+=l2_regularization*0.0005
            loss.backward()
            optimizer.step()

            running_loss+=loss.item()
            count+=16
        with torch.no_grad():
            y_pred_train=mlp.forward(torch.Tensor(x_train))
            y_pred_valid=mlp.forward(torch.Tensor(x_valid))
            train_acc.append((1-zero_one_loss(((y_pred_train>0.5).float()*1).numpy(),y_train)))
            valid_acc.append((1-zero_one_loss(((y_pred_valid > 0.5).float() * 1).numpy(), y_valid)))
        print(f'Epoch: {epoch}, Loss:{(running_loss/count)}')
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(valid_acc, label='Validation Accuracy')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Correct classification probability')
    plt.show()
    plt.figure()
    plt.imshow(mlp.linear1.weight.detach().numpy())
    plt.show()
    print(mlp.linear1.weight)
    print(mlp.linear1.bias)