import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

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

class MLP_1(nn.Module):
    def __init__(self):
        super(MLP_1, self).__init__()
        self.linear1=nn.Linear(784,128)
        self.linear2=nn.Linear(128,10)

    def forward(self,x):
        x=self.linear1(x)
        x=F.relu(x)
        x=self.linear2(x)
        return x

class MLP_2(nn.Module):
    def __init__(self):
        super(MLP_2, self).__init__()
        self.linear1=nn.Linear(784,48)
        self.linear2=nn.Linear(48,10)

    def forward(self,x):
        x=self.linear1(x)
        x=F.relu(x)
        x=F.dropout(x,0.2)
        x=self.linear2(x)
        return x

def setting_1():
    mlp_1 = MLP_1()
    count = 0
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(mlp_1.parameters(), lr=1e-3)
    losses = []
    accuracy = []
    print(mlp_1.linear1.weight.detach().numpy().shape)
    for epoch in range(40):
        correct = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            count += 1
            input = images.view(-1, 28 * 28)
            l1_regularization = torch.tensor(0.)
            l2_regularization = torch.tensor(0.)

            outputs = mlp_1.forward(input)

            # for param in mlp_1.parameters():
            #     # print(param)
            #     l2_regularization+=torch.norm(param,2)**2
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            predictions = torch.max(outputs, 1)[1]
            correct += (predictions == labels).sum().numpy()
        losses.append(loss.data)
        accuracy.append(100 * correct / len(train_loader.dataset))
        print(
            f'Epoch {epoch + 1:02d}, Iteration: {count:5d}, Loss: {loss.data:.4f}, Accuracy: {100 * correct / len(train_loader.dataset):2.3f}%')

    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train Loss Graph')
    plt.show()
    plt.figure()
    plt.plot(accuracy)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy Graph')
    plt.show()
    plt.figure()

    plt.hist(mlp_1.linear1.weight.detach().numpy().T.flatten())
    plt.title('Weight Layer 1 Histogram (No Reg)')
    plt.show()
    plt.figure()
    plt.hist(mlp_1.linear2.weight.detach().numpy().T.flatten())
    plt.title('Weight Layer 2 Histogram (No Reg)')
    plt.show()

def setting_2():
    mlp_2 = MLP_2()
    count = 0
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(mlp_2.parameters(), lr=1e-3)
    losses = []
    accuracy = []
    for epoch in range(40):
        correct = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            count += 1
            input = images.view(-1, 28 * 28)
            l1_regularization = torch.tensor(0.)
            l2_regularization = torch.tensor(0.)

            outputs = mlp_2.forward(input)

            for param in mlp_2.parameters():
                # print(param)
                l2_regularization+=torch.norm(param,2)**2
            loss = loss_func(outputs, labels)+0.0001*l2_regularization
            loss.backward()
            optimizer.step()
            predictions = torch.max(outputs, 1)[1]
            correct += (predictions == labels).sum().numpy()
        losses.append(loss.data)
        accuracy.append(100 * correct / len(train_loader.dataset))
        print(accuracy)
        print(
            f'Epoch {epoch + 1:02d}, Iteration: {count:5d}, Loss: {loss.data:.4f}, Accuracy: {100 * correct / len(train_loader.dataset):2.3f}%')

    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train Loss Graph')
    plt.show()
    plt.figure()
    plt.plot(accuracy)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy Graph')
    plt.show()
    plt.figure()

    plt.hist(mlp_2.linear1.weight.detach().numpy().T.flatten())
    plt.title('Weight Layer 1 Histogram (L2 Reg)')
    plt.show()
    plt.figure()
    plt.hist(mlp_2.linear2.weight.detach().numpy().T.flatten())
    plt.title('Weight Layer 2 Histogram (L2 Reg)')
    plt.show()
if __name__=='__main__':
    # setting_1()

    setting_2()


