from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as nbh
from sklearn.neighbors import DistanceMetric
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from sklearn import preprocessing
import numpy as np


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
    total_nbh = []
    total_scores = []
    neigh = nbh()
    classifier = neigh.fit(features_encoded, targets_encoded)
    print(features_encoded)

    for i in features_encoded:
        for j in range(len(i)):
            temp = i.copy()
            temp[j] = 0





glass()