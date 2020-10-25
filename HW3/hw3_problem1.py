import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm, trange

class Linear():
    def __init__(self, in_features, out_features):
        self.W=np.random.randn(in_features,out_features)*np.sqrt(2/out_features)
        self.b=np.zeros(out_features)
    def __call__(self, x, *args, **kwargs):
        return x@self.W+self.b

class MLP():
    def __init__(self, train_file, test_file):
        self.data={}
        with h5py.File(train_file, 'r') as hf:
            self.data['x_train']=hf['xdata'][:]
            self.data['x_valid']=self.data['x_train'][50000:]
            self.data['x_train']=self.data['x_train'][:50000]
            self.data['y_train']=hf['ydata'][:]
            self.data['y_valid']=self.data['y_train'][50000:]
            self.data['y_train']=self.data['y_train'][:50000]
        with h5py.File(test_file,'r') as hf:
            self.data['x_test']=hf['xdata'][:]
            self.data['y_test']=hf['ydata'][:]
        print('Train set shapes:', self.data['x_train'].shape,self.data['y_train'].shape)
        print('Valid set shapes:', self.data['x_valid'].shape, self.data['y_valid'].shape)
        print('Test set shapes:',self.data['x_test'].shape, self.data['y_test'].shape)

        self.linear1=Linear(in_features=784, out_features=256)
        self.linear2=Linear(in_features=256,out_features=64)
        self.linear3=Linear(in_features=64, out_features=32)
        self.linear4=Linear(in_features=32, out_features=10)

    def relu(self, x):
        return np.clip(x,a_min=0,a_max=None)

    def relu_backward(self,x):
        x[x<=0]=0
        x[x>0]=1
        return x

    def tanh(self,x):
        return np.tanh(x)

    def tanh_backward(self, x):
        out = [1-np.tanh(sample)**2 for sample in x]

        return np.array(out)

    def softmax(self, x):
        out=[np.exp(sample)/np.sum(np.exp(sample)) for sample in x]
        return np.array(out)

    def train(self, epochs, batch_size=50, lr=1e-3):
        losses=[]
        train_accuracy=[]
        valid_accuracy=[]
        test_accuracy=[]
        epochs=tqdm(range(epochs))
        lr_steps=0

        for i in epochs:
            for j in range(0,self.data['x_train'].shape[0],batch_size):

                x_batch=self.data['x_train'][j:j+batch_size]
                # x_batch=(x_batch-np.mean(x_batch))/np.std(x_batch)
                x_batch=x_batch/255.
                noise = np.random.normal(0, .05, x_batch.shape)
                x_batch=x_batch+noise
                y_batch=self.data['y_train'][j:j+batch_size]

                h_1=self.linear1(x_batch)
                a_1=self.relu(h_1)
                h_2=self.linear2(a_1)
                a_2=self.relu(h_2)
                h_3=self.linear3(a_2)
                a_3=self.relu(h_3)
                h_4=self.linear4(a_3)
                a_4=self.softmax(h_4)

                loss=-np.sum(y_batch*np.log(a_4))/y_batch.shape[0]
                losses.append(loss)
                epochs.set_description('%d Loss:%.5f'%(j,loss))

                delta_L=(a_4-y_batch)/x_batch.shape[0]
                da_3 = self.relu_backward(h_3)
                da_2 = self.relu_backward(h_2)
                da_1 = self.relu_backward(h_1)

                delta_3 = np.multiply(da_3, delta_L@self.linear4.W.T)
                delta_2 = np.multiply(da_2,delta_3@self.linear3.W.T)
                delta_1 = np.multiply(da_1,delta_2@self.linear2.W.T)

                # print(np.multiply(delta_L, a_3).shape)
                # print(a_3.shape, y_batch.shape, delta_L.shape, a_3.shape, self.linear3.W.shape, a_2.shape, a_1.shape, delta_2.shape, self.linear2.W.shape)
                self.linear4.W = self.linear4.W - lr* (a_3.T@delta_L)/x_batch.shape[0]
                self.linear4.b = self.linear4.b - lr* np.average(delta_L, axis=0)

                self.linear3.W = self.linear3.W - lr * (a_2.T@delta_3)/x_batch.shape[0]
                self.linear3.b = self.linear3.b - lr * np.average(delta_3,axis=0)

                self.linear2.W = self.linear2.W - lr * (a_1.T@delta_2)/x_batch.shape[0]
                self.linear2.b = self.linear2.b - lr* np.average(delta_2,axis=0)

                self.linear1.W = self.linear1.W - lr * (x_batch.T@delta_1)/x_batch.shape[0]
                self.linear1.b = self.linear1.b - lr * np.average(delta_1,axis=0)
                # break
            # break
            # return
            y_pred_train=self.softmax(self.linear4(self.relu(self.linear3(self.relu(self.linear2(self.relu(self.linear1(self.data['x_train']/255.))))))))
            y_pred_train = np.argmax(y_pred_train, axis=1)
            y_gt_train = np.argmax(self.data['y_train'], axis=1)
            train_accuracy.append((y_pred_train == y_gt_train).sum() / y_gt_train.shape[0])
            y_pred_valid=self.softmax(self.linear4(self.relu(self.linear3(self.relu(self.linear2(self.relu(self.linear1(self.data['x_valid']/255.))))))))
            y_pred_valid=np.argmax(y_pred_valid,axis=1)
            y_gt_valid=np.argmax(self.data['y_valid'],axis=1)
            valid_accuracy.append((y_pred_valid==y_gt_valid).sum()/y_gt_valid.shape[0])
            y_pred_test=self.softmax(self.linear4(self.relu(self.linear3(self.relu(self.linear2(self.relu(self.linear1(self.data['x_test']/255.))))))))
            y_pred_test = np.argmax(y_pred_test, axis=1)
            y_gt_test=np.argmax(self.data['y_test'],axis=1)
            test_accuracy.append((y_pred_test == y_gt_test).sum() / y_gt_test.shape[0])

            print(valid_accuracy[-1], test_accuracy[-1])
            # print(y_pred_valid.shape)
            # print(self.data['y_valid'].shape)

            if i==23:
                lr=lr/2
            if i==37:
                lr=lr/2
            # if np.std(valid_accuracy[-5:])<1e-2 and i>10 and lr_steps<2:
            #     print('Halving LR')
            #     lr=lr/2
            #     lr_steps+=1
            # if np.std(valid_accuracy[-5:])<1e-4 and i>20 and lr_steps<2:
            #     print('Halving LR')
            #     lr=lr/2
            #     lr_steps+=1
        plt.plot(train_accuracy, label='Train accuracy')
        plt.plot(valid_accuracy, label='Valid accuracy')
        plt.plot(test_accuracy, label='Test accuracy')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()
if __name__=='__main__':
    mlp=MLP('mnist_traindata.hdf5','mnist_testdata.hdf5')
    mlp.train(epochs=50, lr=5e-1)

    # linear1=Linear(10,2)
    # print(linear1(np.random.rand(5000,10)).shape)