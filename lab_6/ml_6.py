from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier


def make_plot(scores_list, point_list, ttl, lgn):
    figure = plt.figure()
    plt.plot(scores_list, point_list, label=lgn)
    plt.title(ttl)
    plt.legend()
    #figure.savefig(lgn+'.png')


def ada_grad_ans():
    info = pd.read_csv('vehicle.csv', sep = ',')
    data = info.drop('Class', 1)
    targets = info['Class'].values
    print('Ada boost classifier:')
    for cn, i in enumerate([GaussianNB(), None, DecisionTreeClassifier(max_depth=3)]):
        temp_ada = []
        labl = ''
        if cn == 0:
            labl = 'Bayes'
        elif cn == 1:
            labl = 'DecisionTreeClassifier, max_depth = 1'
        else:
            labl = 'DecisionTreeClassifier, max_depth = 3'
        for j in [10, 20, 60, 80, 100, 1000]:
            ada_booster = AdaBoostClassifier(base_estimator=i, n_estimators=j, learning_rate=1.0, random_state=0)
            ada_booster.fit(data, targets)
            print('Current accuracy: {0}, current estimator: {1}, current n_estimators: {2}'.format(ada_booster.score(data, targets), labl, j))
            temp_ada.append(ada_booster.score(data, targets))
        make_plot(temp_ada, [10, 20, 60, 80, 100, 1000], 'Ada boost Classisifier', labl)

    print('Gradient Boosting Classifier')
    for i in range(1, 4):
        temp_gradient = []
        for est in [10, 20, 60, 80, 100, 1000]:
            grad_boost = GradientBoostingClassifier(max_depth=i, n_estimators=est, random_state=1)
            grad_boost.fit(data, targets)
            temp_gradient.append(grad_boost.score(data, targets))
            print('Current accuracy: {0}, current max_depth = {1}, current n_estimators: {2}'.format(grad_boost.score(data, targets), i, est))
        make_plot(temp_gradient, [10, 20, 60, 80, 100, 1000], 'Gradient boost classifier', 'Gradient Boost Max_depth' + str(i))


def glass():
    info = pd.read_csv('glass.csv', sep=',')
    data = info.drop('Type', 1)
    res = info['Type'].values
    print('Bagging Classifier')
    for cn, i in enumerate([GaussianNB(), None, DecisionTreeClassifier(max_depth=3)]):
        temp_bag = []
        lbl = ''
        if cn == 0:
            lbl = 'Bagging Classifier GaussianNB'
        elif cn == 1:
            lbl = 'Bagging Classifier Decision tree default'
        else:
            lbl = 'Bagging Classifier Decision tree max_depth = 3'
        for j in [10, 20, 60, 80, 100, 1000]:
            bgc = BaggingClassifier(base_estimator=i, n_estimators=j)
            bgc.fit(data, res)
            temp_bag.append(bgc.score(data, res))
            print('Current accuracy {0}, current estimator: {1}, current n_estimators {2}'.format(bgc.score(data, res), lbl, j))
        make_plot(temp_bag, [10, 20, 60, 80, 100, 1000], 'Bagging Classifier', lbl)

    print('Random Forest Classifier')
    for i in range(1, 4):
        temp_forest = []
        for j in [10, 20, 60, 80, 100, 1000]:
            forest = RandomForestClassifier(max_depth=i, n_estimators=j)
            forest.fit(data, res)
            temp_forest.append(forest.score(data, res))
            print('Current accuracy: {0}, current max_depth: {1}, current n_estimators: {2}'.format(forest.score(data, res), i, j))
        make_plot(temp_forest, [10, 20, 60, 80, 100, 1000], 'Random Forest Classifier', 'Random Forest Classifier max depth' + str(i))



ada_grad_ans()
glass()
plt.show()