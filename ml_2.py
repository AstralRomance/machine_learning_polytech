from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as nbh
from sklearn.neighbors import DistanceMetric
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from sklearn import preprocessing
import numpy as np


def make_plot(ratios, accuracies, title):
    plt.figure()
    plt.plot(ratios, [acc[0] for acc in accuracies], label='test data')
    plt.plot(ratios, [acc[1] for acc in accuracies], label='train data')
    plt.xlabel('training data size')
    plt.ylabel('accuracy')
    plt.title(f'{title}\naccuracy(training data size)')
    plt.legend()
    plt.savefig(f'{title}.png')


def accuracy(feat, train, tr_size):
    tr_size = 1-tr_size
    x_test, x_targ, y_test, y_targ = \
        train_test_split(feat,train, test_size=tr_size, random_state=1)
    neigh = nbh(n_neighbors=3, n_jobs=-1)
    neigh.fit(x_targ, y_targ)
    neigh.predict_proba(x_targ)
    return (metrics.accuracy_score(y_test, neigh.predict(x_test)),
            metrics.accuracy_score(y_targ, neigh.predict(x_targ)))


def tic_tac_toe():
    features, targets = [], []
    with open("Tic_tac_toe.txt") as inp:
        for line in inp:
            features.append(line.split(',')[0:9])
            targets.append(line.split(',')[9].strip())
    le = preprocessing.LabelEncoder()
    features_encoded = [le.fit_transform(sample) for sample in features]
    targets_encoded = le.fit_transform(targets)
    ratios = np.linspace(0.01, 0.9, 100)
    accuracies = [accuracy(features_encoded, targets_encoded, ratio) for ratio in ratios]
    make_plot(ratios, accuracies, 'tic-tac-toe')


def spam():
    df = pd.read_csv('spam.csv', sep=',')
    features = df.iloc[:, 1:58].values
    targets = df['type'].values
    targets_encoded = preprocessing.LabelEncoder().fit_transform(targets)
    ratios = np.linspace(0.001, 0.9, 100)
    accuracies = [accuracy(features, targets_encoded, ratio) for ratio in ratios]
    make_plot(ratios, accuracies, 'spam')


def glass():
    df = pd.read_csv('glass.csv', sep=',')
    df = df.drop('Id', 1)
    features = df.drop('Type', 1).values
    targets = df['Type'].values
    le = preprocessing.LabelEncoder()
    features_encoded = [le.fit_transform(sample) for sample in features]
    targets_encoded = le.fit_transform(targets)
    nbhrs = []
    scores = []
    for i in ('euclidean', 'manhattan', 'chebyshev', 'minkowski'):
        for j in range(2, 30):
            neigh = nbh(n_neighbors=j, metric=i, n_jobs=-1)
            classifier = neigh.fit(features_encoded, targets_encoded)
            print('Metric: {0}, n_neighbors: {1} - Score: {2}'.format(i, j ,classifier.score(features_encoded, targets_encoded)))
            nbhrs.append(j)
            scores.append(1-classifier.score(features_encoded, targets_encoded))
        print('scores val:')
        for k in scores:
            print(k)
        plt.figure()
        plt.plot(nbhrs, scores)
        plt.title(i)
        nbhrs = []
        scores = []
    pred = neigh.predict([[1.516, 11.7, 1.01, 1.19, 72.59, 0.43, 11.44, 0.02, 0.1]])
    print('Predicted type: {0}'.format(pred))


tic_tac_toe()
spam()
glass()
plt.show()