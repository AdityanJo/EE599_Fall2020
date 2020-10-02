import numpy as np
import os
import h5py
import matplotlib.pyplot as plt

class MLP():
    def __init__(self, weights_file, testdata_file):
        with h5py.File(weights_file,'r') as hf:
            self.params={
                'W1':hf['W1'][:],
                'b1':hf['b1'][:],
                'W2':hf['W2'][:],
                'b2':hf['b2'][:],
                'W3':hf['W3'][:],
                'b3':hf['b3'][:]
            }
        assert self.params['W1'].shape==(200,784), 'Error in W1 shape'
        assert self.params['b1'].shape==(200,), 'Error in b1 shape'
        assert self.params['W2'].shape==(100,200), 'Error in W2 shape'
        assert self.params['b2'].shape==(100,), 'Error in b2 shape'
        assert self.params['W3'].shape==(10,100), 'Error in W3 shape'
        assert self.params['b3'].shape==(10,), 'Error in b3 shape'

        with h5py.File(testdata_file,'r') as hf:
            self.x_test=hf['xdata'][:]
            self.y_test=hf['ydata'][:]

        assert self.x_test.shape==(10000,784), 'X data shape incorrect'
        assert self.y_test.shape==(10000,10), 'Y data shape incorrect'

    def relu(self, x):
        return np.clip(x,a_min=0,a_max=None)

    def leaky_relu(self,x):
        return np.where(x>0, x, x*0.01)

    def softmax(self, x):
        return np.exp(x)/np.sum(np.exp(x))

    def forward(self, input):
        x=self.relu(np.matmul(self.params['W1'],input.T)+self.params['b1'])
        x=self.relu(np.matmul(self.params['W2'],x.T)+self.params['b2'])
        x=self.softmax(np.matmul((self.params['W3']),x.T)+self.params['b3'])
        return x

    def run_test(self):
        outputs=np.array([self.forward(data) for data in self.x_test])
        pred_class=np.argmax(outputs,axis=-1)
        gt_class=np.argmax(self.y_test,axis=-1)

        tally=np.where(pred_class==gt_class, 1, 0)
        print(np.count_nonzero(tally))

        correct_samples=np.where(tally==1)[0][:5]
        incorrect_samples=np.where(tally==0)[0][:5]

        fig=plt.figure()
        axes=[]
        for i in range(5*2):
            if i<5:
                axes.append(fig.add_subplot(2,5,i+1))
                subplot_title=('Correct:%d, Pred:%d'%(gt_class[correct_samples[i]],pred_class[correct_samples[i]]))
                axes[-1].set_title(subplot_title)
                plt.imshow(np.reshape(self.x_test[correct_samples[i]],(28,28)))
            else:
                axes.append(fig.add_subplot(2, 5, i+1))
                subplot_title = ('Correct:%d, Pred:%d'%(gt_class[incorrect_samples[5-i]],pred_class[incorrect_samples[5-i]]))
                axes[-1].set_title(subplot_title)
                plt.imshow(np.reshape(self.x_test[incorrect_samples[5-i]],(28,28)))
        plt.show()

if __name__=='__main__':
    weights_file='mnist_network_params.hdf5'
    testdata_file='mnist_testdata.hdf5'
    mlp=MLP(weights_file, testdata_file)
    mlp.run_test()