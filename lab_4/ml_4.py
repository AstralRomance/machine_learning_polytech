import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd
from sklearn import preprocessing, metrics


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def make_subplot(ax, clf, x, y, title=None):
    xx, yy = make_meshgrid(x[:, 0], x[:, 1])
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title(title)


def make_subplot2(ax, clf, x1, y1, x2, y2, title=None):
    make_subplot(ax, clf, x1, y1, title)
    ax.scatter(x2[:, 0], x2[:, 1], c=y2, cmap=plt.cm.coolwarm, s=20, edgecolors='k')


def make_plot1(clf, x, y, title=None):
    fig, sub = plt.subplots(1, 1)
    make_subplot(sub, clf, x, y, title)
    return sub


def make_plot2(clf, x1, y1, x2, y2, title=None):
    sub = make_plot1(clf, x1, y1, title)
    sub.scatter(x2[:, 0], x2[:, 1], c=y2, cmap=plt.cm.coolwarm, s=20, edgecolors='k')


def getData(filename, y_encoder):
    df1 = pd.read_csv(filename, delim_whitespace=True)
    x = df1[['X1', 'X2']].values
    y = y_encoder.fit_transform(df1['Color'].values)
    return x, y


def svm_point1():
    le = preprocessing.LabelEncoder()
    data_train, res_train = getData('svmdata1.txt', le)
    data_test, res_test = getData('svmdata1test.txt', le)
    clf = SVC(kernel='linear', C=1.0)
    clf.fit(data_train, res_train)
    make_plot2(clf, data_train, res_train, data_test, res_test)
    print('number of support vectors: ', clf.n_support_)
    print('train accuracy: ', metrics.accuracy_score(res_train, clf.predict(data_train)))
    print('test accuracy: ', metrics.accuracy_score(res_test, clf.predict(data_test)))


def svm_point2():
    le = preprocessing.LabelEncoder()
    data_train, res_train = getData('svmdata2.txt', le)
    data_test, res_test = getData('svmdata2test.txt', le)
    test_count = 0
    train_count = 0
    for i in range(1, 500):
        clf = SVC(kernel='linear', C=i)
        clf.fit(data_train, res_train)
        if metrics.accuracy_score(res_train, clf.predict(data_train)) >= 1.0:
            train_count +=1
            print(i)
        if metrics.accuracy_score(res_test, clf.predict(data_test)) >= 1.0:
            test_count +=1
            print(i)
    print('test dataset {0}'.format(test_count))
    print('train dataset {0}'.format(train_count))


def svm_point3():
    le = preprocessing.LabelEncoder()
    data_train, res_train = getData('svmdata3.txt', le)
    data_test, res_test = getData('svmdata3test.txt', le)
    for i in ['poly', 'rbf', 'sigmoid']:
            for j in range(1, 5):
                clf = SVC(kernel=i, C=1.0, gamma='auto', degree=j)
                clf.fit(data_train, res_train)
                print('current accuracy: {0}; current kernel: {1}; current degree: {2}'.format(metrics.accuracy_score(res_train,
                                                                                                                   clf.predict(data_train)), i, j))

def svm_point4():
    le = preprocessing.LabelEncoder()
    data_train, res_train = getData('svmdata4.txt', le)
    data_test, res_test = getData('svmdata4test.txt', le)
    for i in ['poly', 'rbf', 'sigmoid']:
            for j in range(1, 5):
                clf = SVC(kernel=i, C=1.0, gamma='auto', degree=j)
                clf.fit(data_train, res_train)
                print('current accuracy: {0}; current kernel: {1}; current degree: {2}'.format(metrics.accuracy_score(res_test,
                                                                                                                   clf.predict(data_test)), i, j))

def svm_point5():
    le = preprocessing.LabelEncoder()
    data_train, res_train = getData('svmdata5.txt', le)
    data_test, res_test = getData('svmdata5test.txt', le)

    gammas = [0.1, 1, 10, 100, 1000, 10000]

    for i in ['poly', 'rbf', 'sigmoid']:
        fig, sub = plt.subplots(2, 3)
        for d, j in enumerate(gammas):
            clf = SVC(kernel=i, gamma=j)
            clf.fit(data_train, res_train)
            print('current accuracy: {0}; current kernel: {1}; current gamma: {2}'.format(
                metrics.accuracy_score(res_test, clf.predict(data_test)), i, j))
            ax = sub.flatten()[d]
            make_subplot2(ax, clf, data_train, res_train, data_test, res_test, 'kernel={0}, gamma={1}'.format(i, j))

print('SVM 1\n')
#svm_point1()
print('SVM 2\n')
#svm_point2()
print('SVM 3\n')
#svm_point3()
print('SVM 4\n')
svm_point4()
print('SVM 5\n')
#svm_point5()
plt.show()