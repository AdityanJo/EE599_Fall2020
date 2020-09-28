import numpy as np

class MLP():
    def __init__(self):
        self.W1=np.array([
            [1,-2],
            [3,4]
        ])
        self.W2=np.array([
            [2,2],
            [3,-3]
        ])
        self.b1=np.array([1,0]).T
        self.b2=np.array([0,-4]).T

    def relu(self, arr):
        return np.clip(arr,a_min=0,a_max=None)
    def forward(self, x):
        a1=self.relu(np.matmul(self.W1,x.T)+self.b1)
        a2=np.matmul(self.W2,a1.T)+self.b2
        return a2

if __name__=='__main__':
    x=np.array([1,-1]).T
    net=MLP()
    out=net.forward(x)
    print(out)