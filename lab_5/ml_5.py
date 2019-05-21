from sklearn.cluster import KMeans
#import numpy as np
import pandas as pd
#import matplotlib
from matplotlib import pyplot as plt
import matplotlib
from sklearn import metrics
from sklearn.datasets import make_blobs
#import scipy
#from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram


def make_plot(my_data, classifier):
    colormap = matplotlib.pyplot.cm.rainbow
    norm = matplotlib.colors.Normalize(vmin=0, vmax=4)
    axes = pd.plotting.scatter_matrix(my_data, color=colormap(norm(classifier.labels_)))


def pluton():
    data = pd.read_csv('pluton.csv')
    for i in [10, 20, 30]:
        kmeans_classifier = KMeans(n_clusters=3, random_state=0, max_iter=i).fit(data)
        make_plot(data, kmeans_classifier)
        # оценивается комактность кластеров: чем объекты ближе друг к другу, тем лучше разделение.
        # Компактность оценивается по расстоянию от точек кластера до центроидов, разделимость на расстоянии от центроид кластеров до глобального центроида
        print('current accuracy calinski-harabaz: {0}, current max_iter: {1}'.format(metrics.calinski_harabaz_score(data, kmeans_classifier.labels_), i))
        # Чем меньще значение, тем лучше разбиение
        try:
            d_b_s = metrics.davies_bouldin_score(data, kmeans_classifier.labels_)
        except RuntimeWarning:
            print('runtime warning')
        print('current accuracy davies-bouldin: {0}, current max_iter: {1}'.format(d_b_s, i))


def blobs_generators():
    centers = [[5, 4], [10, 4.5], [2, 6]]
    X, y = make_blobs(n_samples=300, n_features=2, centers=centers, cluster_std=0.2, random_state=5)

    for i in range(X.shape[0]):
        distance_to_center = X[i][y[i]%2] - centers[y[i]][y[i]%2]
        X[i][y[i]%2] += 10 * distance_to_center

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title("ground truth")

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X)

    from lab_5 import kmedoids
    from sklearn.metrics.pairwise import pairwise_distances
    from sklearn.metrics import homogeneity_score
    from sklearn.metrics import completeness_score

    for metric in ['euclidean', 'manhattan']:

        D = pairwise_distances(X, metric=metric)
        clusters, medoids = kmedoids.cluster(D, 3)

        color_dict = {medoids[0]: 0, medoids[1]: 1, medoids[2]: 2}
        clusters = [color_dict[y] for y in clusters]
        print(f'k-means, metric = {metric}')
        print(f'homogeneity-score = {homogeneity_score(y, clusters)}')
        print(f'completeness-score = {completeness_score(y, clusters)}')
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=clusters, marker='o')
        plt.title(f'k-medoids, metric={metric}')


        D_scaled = pairwise_distances(X_scaled, metric=metric)
        clusters, medoids = kmedoids.cluster(D_scaled, 3)

        color_dict = {medoids[0]: 0, medoids[1]: 1, medoids[2]: 2}
        clusters = [color_dict[y] for y in clusters]
        print(f'k-means, metric = {metric}, scaled')
        print(f'homogeneity-score = {homogeneity_score(y, clusters)}')
        print(f'completeness-score = {completeness_score(y, clusters)}')
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=clusters, marker='o')
        plt.title(f'k-medioids, metric={metric}, scaled')


def votes():
    data = pd.read_csv('votes.csv')
    data = data.fillna(0)
    Z = linkage(data, method='ward')
    dendrogram(Z)



#pluton()
blobs_generators()
#votes()
plt.show()