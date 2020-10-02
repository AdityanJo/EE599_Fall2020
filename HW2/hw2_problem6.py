import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP():
    def __init__(self):
        self.W1=np.array([
            [1,-2,1],
            [3,4,-2]
        ])
        self.W2=np.array([
            [1,-2],
            [3,4]
        ])
        self.W3=np.array([
            [2,2],
            [3,-3],
            [2,1]
        ])
        self.b1=np.array([1,-2]).T
        self.b2=np.array([1,0]).T
        self.b3=np.array([0,-4,-2]).T

    def relu(self, arr):
        return np.clip(arr,a_min=0,a_max=None)

    def softmax(self,arr):
        return np.exp(arr)/np.sum(np.exp(arr))

    def forward(self, x):
        a1=self.relu(np.matmul(self.W1,x.T)+self.b1)
        a2=self.relu(np.matmul(self.W2,a1.T)+self.b2)
        a3=self.softmax(np.matmul(self.W3,a2)+self.b3)
        # print(a1)
        # print(a2)
        # print(a3)
        return a3

    def backward(self, x, y):
        pass


class MLP_Autograd(nn.Module):
    def __init__(self):
        super(MLP_Autograd, self).__init__()
        self.linear1=nn.Linear(3,2)
        self.linear2=nn.Linear(2,2)
        self.linear3=nn.Linear(2,3)
    def forward(self,x):
        x=F.relu(self.linear1(x))
        # print(x)
        x=F.relu(self.linear2(x))
        # print(x)
        x=self.linear3(x)
        # print(x)
        # x=F.softmax(x)
        # print(x)
        return x

def softmax(x):
    exp_x=torch.exp(x)
    sum_x=torch.sum(exp_x, dim=1, keepdim=True)
    return exp_x/sum_x

def log_softmax(x):
    return torch.log(torch.exp(x)-torch.sum(torch.exp(x),dim=1,keepdim=True))

def custom_mbce_loss(output, target):
    print(output)
    output=F.sigmoid(output)
    print(output)
    print('log',torch.log(output))
    # print(target*torch.log(output+1e-7))
    return -torch.sum(target*torch.log(output)+(1-target)*torch.log(1-output+1e-7))
if __name__=='__main__':
    # mlp=MLP()
    # mlp.forward(np.array([1,-1,1]))
    # a=torch.Tensor([0.9,0.1])
    # y=torch.Tensor([1,0])
    # print(custom_mbce_loss(a,y))
    # import sys; sys.exit(0)
    mlp_autograd=MLP_Autograd()
    mlp_autograd.zero_grad()

    mlp_autograd.linear1.weight.data=torch.Tensor([[1.,-2.,1.],[3.,4.,-2.]])

    mlp_autograd.linear1.bias.data=torch.Tensor([1.,-2.])
    mlp_autograd.linear2.weight.data=torch.Tensor([[1.,-2.],[3.,4.]])
    mlp_autograd.linear2.bias.data=torch.Tensor([1.,0.])
    mlp_autograd.linear3.weight.data=torch.Tensor([[2.,2.],[3.,-3.],[2.,1.]])
    mlp_autograd.linear3.bias.data=torch.Tensor([0.,-4.,-2.])
    # criterion = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(mlp_autograd.parameters(), lr=0.5)


    y = torch.tensor([[1,1,0]], dtype=torch.float32)
    x = torch.Tensor([[0.,0.,0.]])
    for _ in range(1):
        optim.zero_grad()
        out=mlp_autograd.forward(x)
    # out=out.unsqueeze(0)


        loss=custom_mbce_loss(out,y)
        print(loss.item())
        loss.backward()
        optim.step()
    # print(mlp_autograd)
    # print(mlp_autograd.linear3.weight.grad)
    # print(mlp_autograd.linear3.weight)
    print(mlp_autograd.linear3.bias.grad)
    # print(mlp_autograd.linear3.bias)

    # print(mlp_autograd.linear2.weight.grad)
    # print(mlp_autograd.linear2.weight)
    # print(mlp_autograd.linear2.bias.grad)
    # print(mlp_autograd.linear2.bias)


    # print(mlp_autograd.linear1.weight.grad)
    # print(mlp_autograd.linear1.weight)
    # print(mlp_autograd.linear1.bias.grad)
    # print(mlp_autograd.linear1.bias)
    # with torch.no_grad():
        # print(F.softmax(mlp_autograd.forward(x)))