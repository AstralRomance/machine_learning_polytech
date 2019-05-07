import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def make_meshgrid(x, y, h = 0.2):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min()-1, y.max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_countours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.countourf(xx, yy, Z, **params)
    return out


def svm_point1():
    data_train = []
    res_train = []
    with open('svmdata1.txt') as inpf:
        for i in inpf:
            data_train.append(i.split('\t')[1:3])
            res_train.append(i.split('\t')[3])
    data_train = data_train[1::]
    res_train = res_train[1::]
    res_train = [i.rstrip() for i in res_train]
    clf = SVC(gamma='auto', kernel='linear', C=1.0)
    clf.fit(data_train, res_train)
    data_test = []
    res_target = []
    with open('svmdata1test.txt') as tst_f:
        for i in tst_f:
            data_test.append(i.split('\t')[1:3])
            res_target.append(i.split('\t')[3])
    data_test = data_test[1::]
    res_target = res_target[1::]
    res_target = [i.rstrip() for i in res_target]
    print(data_test)
    print(res_target)
    





svm_point1()
plt.show()