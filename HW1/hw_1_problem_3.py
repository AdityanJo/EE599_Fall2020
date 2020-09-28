import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def subproblem_2(x,h):
    y=np.convolve(x,h)
    # y_prime=np.correlate(x,h,'full')
    # x=np.pad(x,pad_width=1)
    y=[0,0,0,0,0]
    for i in range(len(x)+len(h)-1):
        for j in range(len(h)):
            # print(i)
            # print('=====')
            if i-j<0: continue
            if i-j>=len(x):continue
            # print(x[i-j], h[j])
            y[i]+=x[i-j]*h[j]
    print(y)
    fig=plt.figure()
    plt.stem(y)
    plt.show()
    return y

def subproblem_3(x,h):
    y=signal.correlate(x,h)
    rows=[]
    for row in y:
        if (np.zeros(row.shape[0])==row).all():
            continue
        rows.append(row)
    y_row_crop=np.row_stack(rows)
    cols=[]
    for col in y_row_crop.T:
        if (np.zeros(col.shape[0])==col).all():
            continue
        cols.append(col)
    y_cropped=np.column_stack(cols)
    print(y)
    plt.matshow(y_cropped, cmap='hot')
    plt.show()

if __name__=='__main__':
    x=np.array([1,1,1],dtype=float)
    h=np.array([1,1/2,1/4],dtype=float)
    subproblem_2(x,h)

    x=np.array(
        [
            [0,0,0,0,0,0,0],
            [0,1,1,1,1,1,0],
            [0,1,1,1,1,1,0],
            [0,1,1,1,1,1,0],
            [0,0,0,0,0,0,0]
        ]
    )

    h=np.array(
        [
            [0,0,0,0,0],
            [0,1/4,1/2,1/4,0],
            [0,1/2,1,1/2,0],
            [0,1/4,1/2,1/4,0],
            [0,0,0,0,0]
        ]
    )

    # x_=np.array([
    #     [1,1,1,1,1],
    #     [1,1,1,1,1],
    #     [1,1,1,1,1]
    # ])
    # h=np.array([
    #     [1,2,3],
    #     [4,5,6],
    #     [7,8,9]
    # ])
    # h_=np.array([
    #     [9,8,7],
    #     [6,5,4],
    #     [3,2,1]
    # ])
    subproblem_3(x,h)