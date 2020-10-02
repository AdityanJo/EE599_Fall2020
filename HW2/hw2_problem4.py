import numpy as np
import h5py
import scipy
import matplotlib.pyplot as plt

def subproblem1():
    with h5py.File('binary_random_20fa.hdf5', 'r') as hf:
        human=hf['human'][:]
        machine=hf['machine'][:]
    x_data = np.vstack([human, machine])
    human_corr=np.corrcoef(human.T)
    machine_corr=np.corrcoef(machine.T)
    corr=np.corrcoef(x_data.T)

    return {
        'human_data':human,
        'machine_data':machine,
        'human':human_corr,
        'machine':machine_corr,
        'corr':corr
    }
def subproblem2(R):
    human_R=R['human']
    machine_R=R['machine']
    corr_R=R['corr']
    human_eig_val, human_eig_vec=np.linalg.eig(human_R)
    machine_eig_val, machine_eig_vec=np.linalg.eig(machine_R)
    eig_val, eig_vec = np.linalg.eig(machine_R)
    idx=eig_val.argsort()
    eig_vec=eig_vec[idx[::-1]]
    eig_val=eig_val[idx[::-1]]
    print('Percent of variance captured:', np.sum(eig_val[:2]) / np.sum(eig_val))
    print('Percent of variance captured:',np.sum(human_eig_val[:2])/np.sum(human_eig_val))
    print(np.sum(human_eig_val))
    print(np.sum(machine_eig_val[:2])/np.sum(machine_eig_val))
    print(np.sum(machine_eig_val))
    plt.plot(np.arange(20),human_eig_val, label='Human')
    plt.plot(np.arange(20), machine_eig_val, label='Machine')
    plt.plot(np.arange(20), eig_val, label='Both')
    plt.legend()
    plt.show()

    R['human_eig_val']=human_eig_val,
    R['human_eig_vec']=human_eig_vec,
    R['machine_eig_val']=machine_eig_val,
    R['machine_eig_vec']=machine_eig_vec
    R['eig_vec']=eig_vec
    R['eig_val']=eig_val
    return R

def subproblem3(R):
    from sklearn.metrics import accuracy_score, zero_one_loss
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    human_R=R['human_data']
    machine_R=R['machine_data']
    human_y=-np.ones(5100).T
    machine_y=np.ones(5100).T
    x_data=np.vstack([human_R,machine_R])
    # x_data=np.hstack([x_data])
    import cv2
    print(x_data.shape)

    y_data=np.concatenate([human_y,machine_y])
    print(y_data.shape)

    w=np.zeros(20)
    clf = make_pipeline(SGDClassifier(max_iter=500, tol=1e-3))
    clf.fit(x_data,y_data)
    pred=clf.predict(x_data)
    w = np.linalg.inv(x_data.T @ x_data) @ x_data.T @ y_data
    pred = np.sign(x_data@w)
    print(pred.shape)
    # clf['sgdclassifier'].coef_=np.expand_dims(w,0)
    # pred = clf.predict(x_data)
    print('pred:',pred,y_data)
    print(zero_one_loss(pred,y_data))
    errors_=[]
    # for epoch in range(10):
    #     errors=0
    #     for i,(xi,yi) in enumerate(zip(x_data,y_data)):
    #         print((yi-np.matmul(w,xi.T)))
    #         update=0.001*(yi-np.matmul(w,xi.T))
    #         w+=update*xi
    #         # print(update)
    #         errors+=1 if np.matmul(w,xi.T)>0 and yi<0 else 0
    #         errors += 1 if np.matmul(w, xi.T) < 0 and yi >0 else 0
    #     # print(errors)
    #     errors_.append(errors)
    # plt.plot(list(range(len(errors_))),errors_)
    # plt.show()
    # print(w)
    # print('xt',x_data.T.shape)
    # print(np.matmul(np.linalg.inv(np.matmul(x_data.T,x_data)),x_data.T).shape)
    # w=np.dot(np.matmul(np.linalg.inv(np.matmul(x_data.T,x_data)),x_data.T),y_data)
    # w = x_data.T @ y_data / (x_data.T @x_data)
    # print('w',w.shape)
    print(w)
    # print(np.sum(np.square(y_data - x_data @ w))/10200)
    # print(np.sum(np.square(y_data-np.sign(x_data@w)))/10200)
    # print((np.sign(x_data@w)==y_data).all())
    # print(x_data.shape)
    R['weight']=clf['sgdclassifier'].coef_[0]
    return R


def subproblem4(R):
    random_humans=R['human_data'][np.random.choice(R['human_data'].shape[0],100)]
    random_machines = R['machine_data'][np.random.choice(R['machine_data'].shape[0], 100)]
    print('Largest 2 eigenvalues:', R['eig_val'][:2])
    print(random_humans.shape)
    print(R['eig_vec'][:2].shape)
    proj_humans=R['eig_vec'][:2]@random_humans.T
    proj_machines = R['eig_vec'][:2] @ random_machines.T

    plt.scatter(proj_humans[0],proj_humans[1],c='r')
    plt.scatter(proj_machines[0], proj_machines[1], c='b')

    plt.show()
    print(random_humans.shape)
    return R

def subproblem5(R):
    from sklearn.metrics import zero_one_loss
    from sklearn.linear_model import SGDClassifier, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    human_R = R['human_data']
    machine_R = R['machine_data']
    human_y = -np.ones(5100).T
    machine_y = np.ones(5100).T
    x_data = np.vstack([human_R, machine_R])
    y_data = np.concatenate([human_y, machine_y])
    clf = make_pipeline(LogisticRegression(max_iter=500, tol=1e-3))
    clf.fit(x_data, y_data)
    pred = clf.predict(x_data)
    print(zero_one_loss(pred,y_data))
    print(clf)
    R['classifier']=clf['logisticregression']
    return R
if __name__=='__main__':
    R=subproblem1()
    R=subproblem2(R)
    R=subproblem3(R)
    R=subproblem5(R)
    R=subproblem4(R)
    print(R['corr'])