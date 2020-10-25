import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

train_set=torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_set=torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_loader=torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)
test_loader=torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False)

def output_label(label):
    output_mapping = {
                 0: "T-shirt/Top",
                 1: "Trouser",
                 2: "Pullover",
                 3: "Dress",
                 4: "Coat",
                 5: "Sandal",
                 6: "Shirt",
                 7: "Sneaker",
                 8: "Bag",
                 9: "Ankle Boot"
                 }
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]
output_mapping = {
                 0: "T-shirt/Top",
                 1: "Trouser",
                 2: "Pullover",
                 3: "Dress",
                 4: "Coat",
                 5: "Sandal",
                 6: "Shirt",
                 7: "Sneaker",
                 8: "Bag",
                 9: "Ankle Boot"
                 }
class MLP(nn.Module):
    def __init__(self, dropout_rate):
        super(MLP, self).__init__()
        self.net=nn.Sequential(
            nn.Linear(784,100),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(100,10)
        )

    def forward(self, x):
        x=x.view(100,-1)
        return self.net(x)

if __name__=='__main__':
    mlp=MLP(dropout_rate=0.3)

    running_loss=0.0
    counts=0
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(mlp.parameters(), lr=0.01)
    for i in range(10):
        for x_train, y_train in train_loader:
            optimizer.zero_grad()
            y_pred=mlp(x_train)
            loss=criterion(y_pred,y_train)
            l2_reg=torch.tensor(0.)
            lambda_reg=torch.tensor(0.0001)
            for param in mlp.parameters():
                l2_reg+=torch.norm(param)
            loss+=lambda_reg * l2_reg
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            counts+=1
        corrects=0
        batches=0
        preds=[]
        targets=[]
        for x_test,y_test in test_loader:
            with torch.no_grad():
                y_pred=mlp(x_test)
                preds.append(F.softmax(y_pred).max(1,keepdim=True)[1])
                targets.append(y_test.cpu())
                corrects+=F.softmax(y_pred).max(1,keepdim=True)[1].eq(y_test.view(-1,1)).sum().item()
            batches+=1
        plt.clf()
        plt.figure(figsize=(10,10))
        print('Epoch %d, Loss %f'%(i, running_loss/counts))
        print('Test Accuracy:%f'%(corrects/(batches*100)))
        targets_test=torch.cat(targets).numpy()
        preds_test=(torch.cat(preds)[:,-1]).numpy()
        cf_matrix=confusion_matrix(preds_test,targets_test)

        hm=sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True,
                    fmt='.2%', cmap='Blues', linewidths=.9, xticklabels=list(output_mapping.values()), yticklabels=list(output_mapping.values()))
        hm.get_figure().savefig('epoch_%d_confusion.png'%i)