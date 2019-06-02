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
    figure.savefig(lgn+'.png')


def ada_grad_ans():
    info = pd.read_csv('vehicle.csv', sep = ',')
    data = info.drop('Class', 1)
    data = data[:len(data)//2:]
    data_test = info.drop('Class', 1)
    data_test = data_test[len(data_test)//2::]
    targets = info['Class'].values
    targets = targets[:len(targets)//2:]
    targets_test = info['Class'].values
    targets_test = targets_test[len(targets_test) // 2::]
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
            print('Current accuracy: {0}, current estimator: {1}, current n_estimators: {2}'.format(ada_booster.score(data_test, targets_test), labl, j))
            temp_ada.append(ada_booster.score(data_test, targets_test))
        make_plot([10, 20, 60, 80, 100, 1000], temp_ada,  'Ada boost Classisifier', labl)

    print('Gradient Boosting Classifier')
    for i in range(1, 4):
        temp_gradient = []
        est = [i for i in range(10, 500, 20)]
        for j in est:
            grad_boost = GradientBoostingClassifier(max_depth=i, n_estimators=j, random_state=1)
            grad_boost.fit(data, targets)
            temp_gradient.append(grad_boost.score(data_test, targets_test))
            print('Current accuracy: {0}, current max_depth = {1}, current n_estimators: {2}'.format(grad_boost.score(data_test, targets_test), i, est))
        make_plot(est, temp_gradient, 'Gradient boost classifier', 'Gradient Boost Max_depth' + str(i))


def glass():
    info = pd.read_csv('glass.csv', sep=',')
    data = info.drop('Type', 1)
    data = data[:len(data)//2:]
    data_test = info.drop('Type', 1)
    data_test = data_test[len(data_test)//2::]
    res = info['Type'].values
    res = res[:len(res)//2:]
    res_test = info['Type'].values
    res_test = res_test[len(res_test)//2::]
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
            temp_bag.append(bgc.score(data_test, res_test))
            print('Current accuracy {0}, current estimator: {1}, current n_estimators {2}'.format(bgc.score(data_test, res_test), lbl, j))
        make_plot([10, 20, 60, 80, 100, 1000], temp_bag, 'Bagging Classifier', lbl)

    print('Random Forest Classifier')
    for i in range(1, 4):
        temp_forest = []
        for j in [10, 20, 60, 80, 100, 1000]:
            forest = RandomForestClassifier(max_depth=i, n_estimators=j)
            forest.fit(data, res)
            temp_forest.append(forest.score(data_test, res_test))
            print('Current accuracy: {0}, current max_depth: {1}, current n_estimators: {2}'.format(forest.score(data_test, res_test), i, j))
        make_plot([10, 20, 60, 80, 100, 1000], temp_forest,  'Random Forest Classifier', 'Random Forest Classifier max depth' + str(i))



#ada_grad_ans()
glass()
plt.show()