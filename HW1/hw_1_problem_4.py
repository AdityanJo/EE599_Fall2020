import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def subproblem_1():
    x=np.random.randn(10)
    y=np.zeros(10)
    L=1
    alpha=0.1
    a=[1,-alpha]
    b=[1-alpha]
    for n in range(x.shape[0]):
        sum_1=0
        sum_2=0
        for i in range(L):
            sum_1+=b[i]*x[n-i]
            sum_2+=a[i]*y[n-1]
        y[n]=sum_1-sum_2
    plt.plot(y)
    plt.show()

def subproblem_2(alphas=[0.9,0.5,0.1,-0.5]):
    for alpha in alphas:
        b=1-alpha
        a=[1, -alpha]
        w,h = signal.freqz(b,a)
        v = w/(2*np.pi)

        plt.plot(v,20*np.log10(h),label='alpha=%d'%alpha)
    plt.show()

def subproblem_3(alphas=[0.9,0.5,0.1]):
    def find_closest_to_n(arr, n):
        return min(range(len(arr)), key=lambda x: abs(arr[x]-n))
    for alpha in alphas:
        b=[1-alpha]
        a=[1.0,-alpha]
        imp = signal.unit_impulse(100)
        response = signal.lfilter(b,a,imp)
        idx=find_closest_to_n(response,response[0]*0.2)
        print('Decay to 20% at n=',idx)
        print(response)
        plt.plot(np.arange(0,100),response)
        # plt.show()
    plt.show()

def subproblem_4(l=4, bandwidth=0.25):
    b, a = signal.butter(l,bandwidth)
    print('AR coefficients:',a)
    print('MA coefficients:',b)
    w,h = signal.freqz(b,a)
    v=w/(2*np.pi)
    plt.plot(v,20*np.log10(h))
    plt.show()

def subproblem_5(len=300):
    b,a=signal.butter(4,0.25)
    # samples=[np.random.normal(size=1) for i in range(300)]
    samples=np.random.normal(size=len)
    plt.plot(samples)
    y=signal.lfilter(b,a,samples)
    plt.plot(y)
    plt.show()
subproblem_3()